import time

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torchvision.transforms.functional import InterpolationMode
from timm.models.vision_transformer import PatchEmbed

from .submodule import local_conloss
from .submodule.vit.vit import Block
from functools import partial
from transformers import BertTokenizer

from model.submodule.bert.bert import MyBertMaskedLM
from Utils.pos_embed import get_2d_sincos_pos_embed
# from model.submodule.bert.bert_encoder import BertEncoder
from model.submodule.bert.BertConfig import BertConfig
from torch.autograd import Variable


class MM(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=True, mv=False,
                 temp=0.07, temp1=4.0, temp2=5.0, temp3=10.0,
                 local_contrastive_loss=False,
                 c_embed_dim=256):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(
            './model/submodule/bert/bert-base-uncased')  # Using BERT tokenizer
        # self.idxtoword = [k for k, v in self.tokenizer.get_vocab().items()]
        # self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.local_contrastive_loss = local_contrastive_loss
        if self.local_contrastive_loss:
            self.vision_local_embedding = local_conloss.LocalEmbedding(embed_dim, 2048, embed_dim)
            self.text_local_embedding = local_conloss.LocalEmbedding(embed_dim, 2048, embed_dim)
        # --------------------------------------------------------------------------
        # image encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.temp = nn.Parameter(torch.ones([]) * temp)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.vision_proj = nn.Linear(embed_dim, c_embed_dim)
        self.text_proj = nn.Linear(embed_dim, c_embed_dim)
        # self.itm_head = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim * 2),
        #     nn.LayerNorm(embed_dim * 2),
        #     nn.GELU(),
        #     nn.Linear(embed_dim * 2, 2)
        # )
        self.itm_head = nn.Linear(embed_dim, 2)
        # ViT blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # image decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size * 2) ** 2 * in_chans, bias=True)
        # --------------------------------------------------------------------------
        # Bert encoder
        self.bert_encoder, _ = build_bert()
        # self.bert_mlp = nn.Linear(embed_dim, embed_dim, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.fusion_encoder, _ = build_bert()

        self.mv = mv
        if self.mv:
            self.latent_proj = nn.Linear(embed_dim * 2, embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """

        p = self.patch_embed.patch_size[0] * 2
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0] * 2
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def get_text_embeds(self, text):
        text_output = self.bert_encoder(None, input_ids=text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True)
        text_embeds = text_output.hidden_states[-1]
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        return text_embeds.cuda(), text_feat.cuda(), text_output

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is removed
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_vision_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        hidden_features = []
        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x, i == 11)
            hidden_features.append(x)
        x = self.norm(x)
        hidden_features = torch.stack(hidden_features, dim=0)
        if mask_ratio > 0:
            return x, mask, ids_restore, hidden_features
        else:
            return x, hidden_features

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_vision_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        # print(target.shape, pred.shape)
        loss = (pred - target) ** 2

        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_global_contrastive_loss(self, latent, text_feat: Tensor, temp=0.5):
        latent = F.normalize(self.vision_proj(latent[:, 0, :]), dim=-1)
        c_labels = torch.arange(latent.size(0)).type_as(latent).long()
        scores = latent.mm(text_feat.t()) / temp
        loss = F.cross_entropy(scores, c_labels) + F.cross_entropy(scores.transpose(0, 1), c_labels)
        return loss

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def forward_mlm_loss(self, latent, text):
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, 0.15)

        input_ids, labels = self.mask(input_ids, self.bert_encoder.config.vocab_size, latent.device,
                                      targets=labels,
                                      probability_matrix=probability_matrix)

        image_atts = torch.ones(latent.size()[:-1], dtype=torch.long).to(latent.device)
        outputs = self.bert_encoder(latent=None, input_ids=input_ids, attention_mask=text.attention_mask)
        outputs = self.fusion_encoder(latent=None,
                                      encoder_embeds=outputs.hidden_states[-1],
                                      attention_mask=text.attention_mask,
                                      labels=labels,
                                      encoder_hidden_states=latent,
                                      encoder_attention_mask=image_atts,  # all ones
                                      return_dict=True)
        # ----------------------------
        # another option, original one
        # outputs = self.bert_encoder(latent=latent, input_ids=caption_ids, labels=labels, attention_mask=attention_mask,
        #                           token_type_ids=token_type_ids)

        # print(len(outputs.hidden_states), outputs.hidden_states[0].shape)
        # print(torch.equal(_.last_hidden_state,outputs.hidden_states[-1]))
        return outputs.loss

    def forward_matching_loss(self, v_embed, l_embed, text, l_feat: Tensor):
        """
        :param v_embed: vision embedding
        :param l_embed: text embedding
        :param text: text input
        """
        bs = v_embed.size(0)
        attention_mask = text.attention_mask
        image_atts = torch.ones(v_embed.size()[:-1], dtype=torch.long).to(v_embed.device)
        v_feat = F.normalize(self.vision_proj(v_embed[:, 0, :].contiguous()), dim=-1)
        output_feat = self.fusion_encoder(latent=None,
                                          encoder_embeds=l_embed,
                                          attention_mask=attention_mask,
                                          encoder_hidden_states=v_embed,
                                          encoder_attention_mask=image_atts,  # all ones
                                          return_dict=True)  # fusion module CA
        with torch.no_grad():
            sim_i2t = v_feat @ l_feat.t() / self.temp
            sim_t2i = l_feat @ v_feat.t() / self.temp

            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(v_embed[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(l_embed[neg_idx])
            text_atts_neg.append(attention_mask[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([l_embed, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, v_embed], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.fusion_encoder(latent=None,
                                         encoder_embeds=text_embeds_all,
                                         attention_mask=text_atts_all,
                                         encoder_hidden_states=image_embeds_all,
                                         encoder_attention_mask=image_atts_all,  # all ones
                                         return_dict=True)

        vl_embeddings = torch.cat([output_feat.hidden_states[-1][:, 0, :], output_neg.hidden_states[-1][:, 0, :]],
                                  dim=0)
        vl_output = self.itm_head(vl_embeddings)  # self.itm_head = nn.Linear(text_width, 2)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(
            v_embed.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        return loss_itm

    def forward_local_contrastive_loss(self, img_features, ids, words_emb):
        """
        :param ids: caption_ids from tokenizer
        :param img_features: [b, patch_num, embed]
        :param words_emb: bert output
        :return: loss, attn_maps
        """
        temperature = 0.1
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
        patch_feat = img_features[:, 1:].contiguous()  # [b, patch_num, embed]

        # img_features = img_features.sum(axis=1)  # [b, patch_num, embed]
        # img_features = img_features.permute(0, 2, 1)
        # img_features = img_features / torch.norm(
        #     img_features, 2, dim=1, keepdim=True
        # ).expand_as(img_features)

        # we get img_feat and patch_feat now
        patch_emb = self.vision_local_embedding(patch_feat)
        patch_emb = F.normalize(patch_emb, dim=-1)  # [b, patch_num, embed]

        atten_sim = torch.bmm(word_emb, patch_emb.permute(0, 2, 1))  # [b, words_length, patch_num]
        atten_scores = F.softmax(atten_sim / temperature, dim=-1)  # [b, words_length, patch_num]
        word_atten_output = torch.bmm(atten_scores, patch_emb)  # [b, words_length, embed]
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
        # Do the same thing to word, and sum up at last as local loss!
        atten_sim = torch.bmm(patch_emb, word_emb.permute(0, 2, 1))
        patch_num = patch_emb.size(1)
        mask = torch.from_numpy(np.array(sents)[:, 1:] == "[PAD]").type_as(patch_emb).bool()
        atten_sim[mask.unsqueeze(1).repeat(
            1, patch_num, 1)] = float("-inf")
        atten_scores = F.softmax(
            atten_sim / temperature, dim=-1)  # bz, 196, 111
        patch_atten_output = torch.bmm(atten_scores, word_emb)
        with torch.no_grad():
            img_attn_map = self.blocks[-1].attn.attention_map.detach(
            )
            atten_weights = img_attn_map[:, :, 0, 1:].mean(dim=1)
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

        # batch_size = img_features.shape[0]  # eg. 48
        # words_num = words_emb.shape[2]
        # att_maps = []
        # similarities = []
        # # cap_lens = cap_lens.data.tolist()
        # for i in range(batch_size):
        #     # Get the i-th text description
        #     word = words_emb[i, :, :].unsqueeze(0).contiguous()
        #     word = word.repeat(batch_size, 1, 1)  # [48, 768, 100]
        #     context = img_features  # [48, 768, 196]
        #
        #     weiContext, attn = attention_fn(
        #         word, context, temp1
        #     )  # [48, 768, 100], [48, 100, 196]
        #
        #     att_maps.append(
        #         attn[i].unsqueeze(0).contiguous()
        #     )  # add attention for curr index  [100, 196]
        #     word = word.transpose(1, 2).contiguous()  # [48, 100, 768]
        #     weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 100, 768]
        #
        #     word = word.view(batch_size * words_num, -1)  # [4800, 768]
        #     weiContext = weiContext.view(batch_size * words_num, -1)  # [4800, 768]
        #
        #     row_sim = cosine_similarity(word, weiContext)
        #     row_sim = row_sim.view(batch_size, words_num)  # [48, 100]
        #
        #     row_sim.mul_(temp2).exp_()
        #     row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
        #     row_sim = torch.log(row_sim)
        #
        #     similarities.append(row_sim)
        #
        # similarities = torch.cat(similarities, 1)  #
        # similarities = similarities * temp3
        # similarities1 = similarities.transpose(0, 1).contiguous()  # [48, 48]
        #
        # labels = Variable(torch.LongTensor(range(batch_size))).to(similarities.device)
        #
        # loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
        # loss1 = nn.CrossEntropyLoss()(similarities1, labels)
        return loss_local

    def forward(self, batch, mask_ratio=0.75):
        imgs_1, text = batch['image1'], batch['text']
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        # split different views of images
        imgs_1 = imgs_1.cuda()
        _imgs = torchvision.transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC)(imgs_1)
        text = self.tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(
            imgs_1.device)
        text_embeds, text_feat, text_output = self.get_text_embeds(text)
        latent, mask, ids_restore, _ = self.forward_vision_encoder(_imgs,
                                                                   mask_ratio)  # latent: [N, 50, D], 50=maskratio*196
        latent_unmasked, hidden_features = self.forward_vision_encoder(_imgs, 0.0)  # latent_unmasked: [N, 196, D]
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        v_loss = self.forward_vision_loss(imgs_1, pred, mask)
        if self.mv:
            imgs_2 = batch['image2']
            imgs_2 = imgs_2.cuda()
            _imgs_2 = torchvision.transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC)(imgs_2)
            latent_2, mask_2, ids_restore_2, _ = self.forward_vision_encoder(_imgs_2, mask_ratio)
            latent_unmasked_2, hidden_features_2 = self.forward_vision_encoder(_imgs_2, 0.0)

            latent_unmasked = torch.cat((latent_unmasked, latent_unmasked_2), dim=-1)
            hidden_features = torch.cat((hidden_features, hidden_features_2), dim=-1)
            latent_unmasked = self.latent_proj(latent_unmasked)
            hidden_features = self.latent_proj(hidden_features)

            latent = torch.cat((latent, latent_2), dim=-1)
            latent = self.latent_proj(latent)

            pred_2 = self.forward_decoder(latent_2, ids_restore_2)
            v_loss = 0.5 * v_loss + 0.5 * self.forward_vision_loss(imgs_2, pred_2, mask_2)
        loss = []
        global_contrastive_loss = self.forward_global_contrastive_loss(latent_unmasked, text_feat, self.temp)
        loss.append(v_loss)
        loss.append(global_contrastive_loss)
        if self.local_contrastive_loss:
            # t = time.time()
            local_contrastive_loss = self.forward_local_contrastive_loss(latent_unmasked, text.input_ids, text_output)
            # print('local:', time.time() - t)
            loss.append(local_contrastive_loss)
        mlm_loss = self.forward_mlm_loss(latent, text)
        itm_loss = self.forward_matching_loss(latent_unmasked, text_embeds, text, text_feat)
        loss.append(mlm_loss)
        loss.append(itm_loss)
        return loss, pred, mask


def build_bert():
    bert_config = BertConfig()
    text_encoder = MyBertMaskedLM(bert_config)
    text_width = text_encoder.config.hidden_size
    return text_encoder, text_width


# def attention_fn(query, context, temp1):
#     """
#     query: batch x ndf x sequence_length (ndf is the hidden_embed_size)
#     context: batch x ndf x patch_num
#     :returns
#     weightedContext: batch x ndf x sequence_length
#     attn: batch x sequence_length x patch_num
#     """
#     batch_size, sequence_length = query.size(0), query.size(2)
#     patch_num = context.size(2)
#
#     # --> batch x patch_num x ndf
#     contextT = torch.transpose(context, 1, 2).contiguous()
#
#     # Get attention
#     # (batch x patch_num x ndf)(batch x ndf x sequence_length)
#     # -->batch x patch_num x sequence_length
#     attn = torch.bmm(contextT, query)
#     # --> batch*patch_num x sequence_length
#     attn = attn.view(batch_size * patch_num, sequence_length)
#     attn = nn.Softmax(dim=-1)(attn)
#
#     # --> batch x patch_num x sequence_length
#     attn = attn.view(batch_size, patch_num, sequence_length)
#     # --> batch*sequence_length x patch_num
#     attn = torch.transpose(attn, 1, 2).contiguous()
#     attn = attn.view(batch_size * sequence_length, patch_num)
#
#     attn = attn * temp1
#     attn = nn.Softmax(dim=-1)(attn)
#     attn = attn.view(batch_size, sequence_length, patch_num)
#     # --> batch x patch_num x sequence_length
#     attnT = torch.transpose(attn, 1, 2).contiguous()
#
#     # (batch x ndf x patch_num)(batch x patch_num x sequence_length)
#     # --> batch x ndf x sequence_length
#     weightedContext = torch.bmm(context, attnT)
#
#     return weightedContext, attn


# def cosine_similarity(x1, x2, dim=1, eps=1e-8):
#     """Returns cosine similarity between x1 and x2, computed along dim."""
#     w12 = torch.sum(x1 * x2, dim)
#     w1 = torch.norm(x1, 2, dim)
#     w2 = torch.norm(x2, 2, dim)
#     return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
