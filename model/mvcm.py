import datetime
import logging
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torchvision.transforms import InterpolationMode

from Utils.misc import is_dist_avail_and_initialized, get_world_size, get_rank, MetricLogger
from model.submodule import local_conloss
from functools import partial
from transformers import BertTokenizer
from timm.models.vision_transformer import PatchEmbed
from model.submodule.BLIP import QFormer
from model.submodule.BLIP.BLIPBase import (Blip2Base, disabled_train)
from model.submodule.BLIP.BlipOutput import BlipOutput, BlipOutputFeatures
from model.submodule.vit.vit import Block
import lightning.pytorch as pl


class MVCM(pl.LightningModule, Blip2Base):
    def __init__(self, img_size=224,
                 embed_dim=768,
                 drop_path_rate=0,
                 use_grad_checkpoint=False,
                 vit_precision="fp16",
                 vit_path='',
                 vit_type='eva_vit',
                 bert='bert-base-uncased',
                 distill=False, queue_size=65536, mv=False,
                 freeze_vit=True,
                 local_contrastive_loss=False,
                 c_embed_dim=256, num_query_token=32, cross_attention_freq=2, checkpoint=True, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if not bert.startswith('.'):
            tokenizer_config = os.path.join('./model/submodule/bert/', bert)
        else:
            tokenizer_config = bert

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_config)
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        # VIT encoder part
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(vit_path, img_size, drop_path_rate,
                                                                       use_grad_checkpoint, vit_precision,
                                                                       encoder=vit_type)
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # Loss options
        self.local_contrastive_loss = local_contrastive_loss
        if self.local_contrastive_loss:
            self.vision_local_embedding = local_conloss.LocalEmbedding(self.visual_encoder.num_features, 2048,
                                                                       128)
            self.text_local_embedding = local_conloss.LocalEmbedding(embed_dim, 2048, 128)

        self.mv = mv
        self.distill = distill
        # Qformer part
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq, tokenizer_config=tokenizer_config,
            checkpoint=checkpoint
        )

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, c_embed_dim)  # 768, 256
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, c_embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.temp1 = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = 50

        if self.distill:
            self.vision_proj_m = deepcopy(self.vision_proj)
            self.text_proj_m = deepcopy(self.text_proj)
            self.Qformer_m = deepcopy(self.Qformer)
            self.model_pairs = [[self.Qformer, self.Qformer_m],
                                [self.vision_proj, self.vision_proj_m],
                                [self.text_proj, self.text_proj_m]]
            self.queue_size = queue_size
            self.register_buffer("image_queue", torch.randn(c_embed_dim, num_query_token, queue_size))
            self.register_buffer("text_queue", torch.randn(c_embed_dim, queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
            self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

            self.copy_params()
            self.alpha = 0.4
            self.momentum = 0.995

    def forward_local_contrastive_loss(self, query_output, ids, words_emb, query_output_m=None, words_emb_m=None):
        """
        :param ids: caption_ids from tokenizer
        :param img_features: [b, patch_num==query_num, v_embed]
        :param words_emb: bert output
        :return: loss
        """

        img_features = query_output.last_hidden_state
        temperature = 0.07
        # get the local word embed
        bz = img_features.size(0)
        all_feat = words_emb.hidden_states[-1].unsqueeze(1)  # [b, layer, words_length, embed]
        last_layer_attn = words_emb.attentions[-1][:, :, 0, 1:].mean(dim=1)
        # t = time.time()
        all_feat, sents, word_atten = local_conloss.aggregate_tokens(self, all_feat,
                                                                     ids, last_layer_attn)
        # print("time for aggregate_tokens", time.time() - t)
        word_atten = word_atten[:, 1:].contiguous()
        all_feat = all_feat[:, 0]
        # report_feat = all_feat[:, 0].contiguous()
        word_feat = all_feat[:, 1:].contiguous()  # [b, words_length, embed]
        # we get report_feat, word_feat, last_atten_pt, sents now
        word_emb = self.text_local_embedding(word_feat)
        word_emb = F.normalize(word_emb, dim=-1)
        # words_emb: [b, embed, words_length]

        # same to the image features because they are all transformer based
        # img_feat=img_features[-1, :, 0].contiguous()  # [b, embed]
        patch_feat = img_features.contiguous()  # [b, patch_num, v_embed]

        # img_features = img_features.sum(axis=1)  # [b, patch_num, embed]
        # img_features = img_features.permute(0, 2, 1)
        # img_features = img_features / torch.norm(
        #     img_features, 2, dim=1, keepdim=True
        # ).expand_as(img_features)

        # we get img_feat and patch_feat now
        patch_emb = self.vision_local_embedding(patch_feat)
        patch_emb = F.normalize(patch_emb, dim=-1)  # [b, patch_num, embed=128]

        atten_sim = torch.bmm(word_emb, patch_emb.permute(0, 2, 1))  # [b, words_length, patch_num]
        atten_scores_v = F.softmax(atten_sim / temperature, dim=-1)  # [b, words_length, patch_num]
        word_atten_output = torch.bmm(atten_scores_v, patch_emb)  # [b, words_length, embed]
        word_atten_output = F.normalize(word_atten_output, dim=-1)
        with torch.no_grad():
            atten_weights = word_atten.detach()
            word_atten_weights = []
            for i in range(bz):
                atten_weight = atten_weights[i]
                nonzero = atten_weight.nonzero().squeeze()
                low = torch.quantile(atten_weight[nonzero], 0.1)
                high = torch.quantile(atten_weight[nonzero], 0.9)
                atten_weight[nonzero] = atten_weight[nonzero].clip(low, high)
                word_atten_weights.append(atten_weight.clone())
            word_atten_weights = torch.stack(word_atten_weights)

        word_atten_weights /= word_atten_weights.sum(dim=1, keepdims=True)
        word_sim = torch.bmm(word_emb, word_atten_output.permute(
            0, 2, 1)) / temperature
        word_num = word_sim.size(1)
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")
        targets = torch.arange(word_num).type_as(
            word_emb).long().repeat(bz)
        loss_word_1 = torch.sum(F.cross_entropy(
            word_sim_1, targets, reduction="none") * word_atten_weights.view(-1)) / bz

        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = torch.sum(F.cross_entropy(
            word_sim_2, targets, reduction="none") * word_atten_weights.view(-1)) / bz

        loss_word = (loss_word_1 + loss_word_2) / 2.

        # -------------------------------------------------------------
        # Do the same thing to query, and sum up at last as local loss!
        atten_sim = torch.bmm(patch_emb, word_emb.permute(0, 2, 1))
        patch_num = patch_emb.size(1)
        mask = torch.from_numpy(np.array(sents)[:, 1:] == "[PAD]").type_as(patch_emb).bool()
        atten_sim[mask.unsqueeze(1).repeat(
            1, patch_num, 1)] = float("-inf")
        atten_scores_w = F.softmax(
            atten_sim / temperature, dim=-1)  # bz, 196, 111
        patch_atten_output = torch.bmm(atten_scores_w, word_emb)
        with torch.no_grad():
            # img_attn_map = self.visual_encoder.blocks[-1].attn.attention_map.detach()
            img_attn_map = query_output.attentions[-1].detach()
            # atten_weights = img_attn_map[:,:,0,:].mean(dim=1)
            atten_weights = img_attn_map.max(dim=2)[0].mean(dim=1)
            patch_atten_weights = []
            for i in range(bz):
                atten_weight = atten_weights[i]
                atten_weight = atten_weight.clip_(torch.quantile(
                    atten_weight, 0.1), torch.quantile(atten_weight, 0.9))
                patch_atten_weights.append(atten_weight.clone())
            patch_atten_weights = torch.stack(patch_atten_weights)
        patch_atten_weights /= patch_atten_weights.sum(
            dim=1, keepdims=True)

        patch_sim = torch.bmm(patch_emb, patch_atten_output.permute(
            0, 2, 1)) / temperature
        patch_num = patch_sim.size(1)
        patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
        targets = torch.arange(patch_num).type_as(
            patch_emb).long().repeat(bz)
        # loss_patch_1 = F.cross_entropy(patch_sim_1, targets)
        loss_patch_1 = torch.sum(F.cross_entropy(
            patch_sim_1, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

        patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
        loss_patch_2 = torch.sum(F.cross_entropy(
            patch_sim_2, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

        loss_patch = (loss_patch_1 + loss_patch_2) / 2.

        loss_local = loss_patch + loss_word

        return loss_local, [atten_scores_v, sents]

    def forward(self, batch, batch_idx, mode='train'):
        if mode == 'train':
            return self.train_forward(batch, batch_idx)
        elif mode == 'valid':
            return self.valid_forward(batch)
        elif mode == 'test':
            return self.test_forward(batch)
        else:
            raise ValueError("mode not supported")

    def train_forward(self, batch, batch_idx ):

        self.alpha = self.alpha * min(1, (self.current_epoch * len(self.train_dataloader()) + batch_idx ) / len(self.train_dataloader()))

        image = batch["image1"]
        text = batch["text"]
        loss = {}
        image = image.cuda()
        rank = dist.get_rank()
        bs = image.size(0)
        # imgs_1 = image.clone()  # keep the 448 size image

        # _imgs = torchvision.transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC)(image)
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        image_feats_all = concat_all_gather(
            image_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]

        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        ###============== GLobal Image-text Contrastive ===================###
        if self.distill:
            with torch.no_grad():
                self._momentum_update()
                query_output_m = self.Qformer_m.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    use_cache=True,
                    return_dict=True,
                )
                image_feats_m = F.normalize(
                    self.vision_proj_m(query_output_m.last_hidden_state), dim=-1
                )
                image_feats_all_m = concat_all_gather(
                    image_feats_m
                )
                text_output_m = self.Qformer_m.bert(
                    text_tokens.input_ids,
                    attention_mask=text_tokens.attention_mask,
                    return_dict=True,
                )
                text_feat_m = F.normalize(
                    self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1
                )

                image_feats_q = torch.cat([image_feats_m.permute(2, 1, 0), self.image_queue.clone().detach()],
                                          dim=2)  # [c_embed_dim, num_query, queue_size+bs]
                text_feat_q = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()],
                                        dim=1)  # [c_embed_dim, queue_size+bs]
                sim_i2t_m = torch.einsum('b q d, d z -> b z q', image_feats_m, text_feat_q).max(-1)[
                                0] / self.temp  # [bs, queue_size+bs]
                sim_t2i_m = torch.einsum('b d, d q z -> b z q', text_feat_m, image_feats_q).max(-1)[
                                0] / self.temp  # [bs, queue_size+bs]

                sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                sim_targets.fill_diagonal_(1)

                sim_i2t_targets = self.alpha * F.softmax(sim_i2t_m, dim=1) + (1 - self.alpha) * sim_targets
                sim_t2i_targets = self.alpha * F.softmax(sim_t2i_m, dim=1) + (1 - self.alpha) * sim_targets

            sim_i2t_q = torch.einsum('b q d, d z -> b z q', image_feats, text_feat_q).max(-1)[0] / self.temp
            sim_t2i_q = torch.einsum('b d, d q z -> b z q', text_feat, image_feats_q).max(-1)[0] / self.temp

            loss_i2t = -torch.sum(
                F.log_softmax(sim_i2t_q, dim=1) * sim_i2t_targets, dim=1
            ).mean()
            loss_t2i = -torch.sum(
                F.log_softmax(sim_t2i_q, dim=1) * sim_t2i_targets, dim=1
            ).mean()

            loss_itc = (loss_i2t + loss_t2i) / 2
            loss['itga'] = loss_itc
            self._dequeue_and_enqueue(image_feats_m, text_feat_m)
        else:

            targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
                image.device
            )

            loss_itc = (
                               F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                               + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                       ) / 2
            loss['itga'] = loss_itc

        ###============== Multiview Contrastive ===================###
        if self.mv:
            image2 = batch["image2"]
            image2 = image2.cuda()
            # _imgs2 = torchvision.transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC)(image2)
            image_embeds2 = self.ln_vision(self.visual_encoder(image2))

            query_output2 = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds2,
                encoder_attention_mask=image_atts,
                use_cache=True,
                return_dict=True,
            )

            image_feats2 = F.normalize(
                self.vision_proj(query_output2.last_hidden_state), dim=-1
            )

            image_feats_all2 = concat_all_gather(
                image_feats2
            )
            sim_i1_i2 = torch.einsum('n p d, m q d -> n m p q', image_feats, image_feats_all2)
            sim_i1_i2 = sim_i1_i2.max(-1)[0]
            sim_i1_i2 = sim_i1_i2.max(-1)[0]
            sim_i1_i2 = sim_i1_i2 / self.temp1
            # sim_i2_i1 = torch.matmul(
            #     rearrange(image_feats2, "b n e -> b (n e)"),
            #     rearrange(image_feats_all, "b n e -> b (n e)").t()
            # )
            sim_i2_i1 = torch.einsum('n p d, m q d -> n m p q', image_feats2, image_feats_all)
            sim_i2_i1 = sim_i2_i1.max(-1)[0]
            sim_i2_i1 = sim_i2_i1.max(-1)[0]
            sim_i2_i1 = sim_i2_i1 / self.temp1

            if not self.distill:
                # sim_i1_i2 = torch.matmul(
                #     rearrange(image_feats, "b n e -> b (n e)"),
                #     rearrange(image_feats_all2, "b n e -> b (n e)").t()
                # )
                targets_mv = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(image.device)

                inter_view_loss = (
                                          F.cross_entropy(sim_i1_i2, targets_mv, label_smoothing=0.1)
                                          + F.cross_entropy(sim_i2_i1, targets_mv, label_smoothing=0.1)
                                  ) / 2

            else:
                with torch.no_grad():
                    query_output2_m = self.Qformer_m.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds2,
                        encoder_attention_mask=image_atts,
                        use_cache=True,
                        return_dict=True,
                    )

                    image_feats2_m = F.normalize(
                        self.vision_proj(query_output2_m.last_hidden_state), dim=-1
                    )

                    image_feats_all2_m = concat_all_gather(
                        image_feats2_m
                    )

                    sim_i1_i2_m = torch.einsum('n p d, m q d -> n m p q', image_feats_m, image_feats_all2_m)
                    sim_i1_i2_m = sim_i1_i2_m.max(-1)[0]
                    sim_i1_i2_m = sim_i1_i2_m.max(-1)[0]
                    sim_i1_i2_m = sim_i1_i2_m / self.temp1

                    sim_i2_i1_m = torch.einsum('n p d, m q d -> n m p q', image_feats2_m, image_feats_all_m)
                    sim_i2_i1_m = sim_i2_i1_m.max(-1)[0]
                    sim_i2_i1_m = sim_i2_i1_m.max(-1)[0]
                    sim_i2_i1_m = sim_i2_i1_m / self.temp1

                    sim_targets = torch.zeros(sim_i1_i2_m.size()).to(image.device)
                    sim_targets.fill_diagonal_(1)

                    sim_i1_i2_targets = self.alpha * F.softmax(sim_i1_i2_m, dim=1) + (1 - self.alpha) * sim_targets
                    sim_i2_i1_targets = self.alpha * F.softmax(sim_i2_i1_m, dim=1) + (1 - self.alpha) * sim_targets

                loss_i1_i2 = -torch.sum(
                    F.log_softmax(sim_i1_i2, dim=1) * sim_i1_i2_targets, dim=1
                ).mean()

                loss_i2_i1 = -torch.sum(
                    F.log_softmax(sim_i2_i1, dim=1) * sim_i2_i1_targets, dim=1
                ).mean()

                inter_view_loss = (loss_i1_i2 + loss_i2_i1) / 2

            loss['imva'] = inter_view_loss

        ###============== Image-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        image_embeds_world = all_gather_with_grad(image_embeds)
        with torch.no_grad():
            sim_t2i[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)
            sim_i2t[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)

            max_val = torch.max(sim_i2t, dim=1, keepdim=True)[0]  # for numeric stability
            sim_i2t -= max_val
            max_val = torch.max(sim_t2i, dim=1, keepdim=True)[0]  # for numeric stability
            sim_t2i -= max_val

            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

        # select one negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select one negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # clear middle variables to save memory as more as possible
        del text_ids_neg, text_atts_neg
        # 在计算完 sim_q2t 和 sim_t2q 后，如果这些变量后面不再需要，可以删除
        del sim_q2t, sim_t2q, sim_i2t, sim_t2i, sim_i1_i2, sim_i2_i1
        # 完成 loss_i2t 和 loss_t2i 的计算后
        del sim_i2t_q, sim_t2i_q, sim_i2t_targets, sim_t2i_targets
        # 清理模型并释放未使用的缓存
        torch.cuda.empty_cache()

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)
        loss['itm'] = loss_itm
        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        if self.distill:
            with torch.no_grad():
                logits_m = self.Qformer_m(decoder_input_ids,
                                          attention_mask=attention_mask,
                                          past_key_values=query_output_m.past_key_values,
                                          return_dict=True,
                                          return_logits=True,
                                          )
            itg_output = self.Qformer(
                decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=query_output.past_key_values,
                return_dict=True,
                labels=labels,
                soft_labels=F.softmax(logits_m, dim=-1),
                alpha=self.alpha
            )
        else:
            itg_output = self.Qformer(
                decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=query_output.past_key_values,
                return_dict=True,
                labels=labels,
            )

        loss_itg = itg_output.loss
        loss['itg'] = loss_itg

        ###============== Local Image-text Contrastive ===================###
        if self.local_contrastive_loss:
            loss_local, _ = self.forward_local_contrastive_loss(query_output, text_tokens.input_ids, text_output,
                                                                query_output_m=query_output_m if self.distill else None,
                                                                words_emb_m=text_output_m if self.distill else None)
            loss['itla'] = loss_local

        return loss

    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=3,
            max_length=30,
            min_length=10,
            top_p=0.9,
            repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                    image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                    caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        if dist.is_available() and dist.is_initialized():
            image_feats = concat_all_gather(image_feat)
            text_feats = concat_all_gather(text_feat)
        else:
            image_feats = image_feat
            text_feats = text_feat
        # image_feats = concat_all_gather(image_feat)
        # text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[..., ptr:ptr + batch_size] = image_feats.T
        self.text_queue[..., ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # world_size=1
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    # tensor_all = GatherLayer.apply(tensors)
    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)
