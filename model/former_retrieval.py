import datetime

import einops
import torch
import torch.nn.functional as F
from functools import partial
import torch.nn as nn
from transformers import BertTokenizer

from model.archi_mvcm import Archi_Mvcm
import torch.distributed as dist
from Utils.misc import is_dist_avail_and_initialized, get_world_size, get_rank, MetricLogger

import logging
import time


class Former_Retrieval(Archi_Mvcm):
    """
    BLIP Image-Text Matching (ITM) model.
    Supported model types:
        - pretrained: pretrained model
        - coco: fintuned model on coco
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, vision_width=768,
            embed_dim=768, depth=12, num_heads=12,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            vit_path='',vit_type='eva_vit',
            decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
            mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), mv=False,
            freeze_vit=True,
            bert='bert-base-uncased',
            local_contrastive_loss=False,
            c_embed_dim=256, num_query_token=32, cross_attention_freq=2, **kwargs
    ):
        super().__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, vision_width=vision_width,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            vit_type=vit_type,
            vit_path=vit_path,
            decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio, norm_layer=norm_layer, mv=mv,
            freeze_vit=freeze_vit, bert=f'../model/submodule/bert/{bert}' if not bert.startswith('.') else bert,
            local_contrastive_loss=local_contrastive_loss,
            c_embed_dim=c_embed_dim, num_query_token=num_query_token, cross_attention_freq=cross_attention_freq
        )

    def forward1(self, samples, match_head="itm"):
        image = samples["image"].to(self.device)
        caption = samples["text_input"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        text = self.tokenizer(
            caption,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        if match_head == "itm":
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                image.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
            output_itm = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            itm_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
            itm_logit = self.itm_head(itm_embeddings)
            itm_logit = itm_logit.mean(dim=1)

            return itm_logit

        elif match_head == "itc":
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_feats = F.normalize(
                self.vision_proj(query_output.last_hidden_state), dim=-1
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_feat = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )

            sims = torch.bmm(image_feats, text_feat.unsqueeze(-1))
            sim, _ = torch.max(sims, dim=1)

            return sim

    def forward(self, data_loader, k_test=64):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)


@torch.no_grad()
def compute_sim_matrix(model, data_loader, **kwargs):
    '''
    :param model:
    :param data_loader: no shuffle data loader
    :param kwargs:
    :return:
    '''
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    # # ------------------- query text-------------------
    # num_text = len(texts)
    # text_bs = 32
    #
    # for i in range(0, num_text, text_bs):
    #     text = texts[i: min(num_text, i + text_bs)]
    #     text_input = model.tokenizer(
    #         text,
    #         padding="max_length",
    #         truncation=True,
    #         max_length=35,
    #         return_tensors="pt",
    #     ).to(model.device)
    #     text_feat = model.forward_text(text_input)
    #     text_embed = F.normalize(model.text_proj(text_feat), dim=-1)
    #     text_embeds.append(text_embed)
    #     text_ids.append(text_input.input_ids)
    #     text_atts.append(text_input.attention_mask)
    #
    # text_embeds = torch.cat(text_embeds, dim=0)
    # text_ids = torch.cat(text_ids, dim=0)
    # text_atts = torch.cat(text_atts, dim=0)
    #
    # # ------------------- query image-------------------
    # qimage_embeds = []
    # image_bs = 32
    # num_image = len(images)
    # for i in range(0, num_image, image_bs):
    #     image = torch.stack(images[i: min(num_image, i + image_bs)], dim=0)
    #     image = image.to(model.device)
    #     with model.maybe_autocast():
    #         image_feat, image_embed = model.forward_image(image)
    #     image_embed = F.normalize(model.vision_proj(image_feat), dim=-1)
    #     qimage_embeds.append(image_embed)
    #
    # qimage_embeds = torch.cat(qimage_embeds, dim=0)  # [num_image, embed_dim]

    # ------------------- candidate image-------------------
    vit_feats = []
    image_embeds = []
    text_ids = []
    text_embeds = []
    text_atts = []
    for samples in data_loader:
        image = samples["image"]
        image = image.to(model.device)
        with model.maybe_autocast():
            image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_cross_modal_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat)
        image_embeds.append(image_embed)

        text = samples["text"]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=45,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_cross_modal_proj(text_feat), dim=-1)
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
        # del image, image_feat, vit_feat, image_embed, text_embed, text_feat
        # torch.cuda.empty_cache()
    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)  # [num_candidate, query_num, embed_dim]

    # print('image_embeds:', image_embeds.size(), 'qimage_embeds:', qimage_embeds.size(), 'text_embeds:', text_embeds.size())
    # image_embeds: torch.Size([1600, 32, 256]) qimage_embeds: torch.Size([80, 32, 256]) text_embeds: torch.Size([40, 256])

    # sims_i2i= torch.einsum('n p d, m q d -> n m p q', qimage_embeds, image_embeds)
    # sims_i2i, _ = sims_i2i.max(-1)
    # sims_i2i, _ = sims_i2i.max(-1)
    # sims_i2i = torch.mm(qimage_embeds, image_embeds.t())  # [num_image, num_candidate]

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)   # [num_candidate, num_text]
    length=len(data_loader.dataset)
    score_matrix_i2t = torch.full(
        (length, length), -100.0
    ).to(model.device)

    num_tasks = get_world_size()
    rank = get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim
        # score_matrix_i2t[start + i, topk_idx] = topk_sim
    # score_matrix_t2i = score_matrix_i2t.t()
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (length, length), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    #
    for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim
        # score_matrix_t2i[start + i, topk_idx] = topk_sim

    if is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    ret = {
        'i2t': F.softmax(score_matrix_i2t, dim=-1).cpu().numpy(),
        't2i': F.softmax(score_matrix_t2i, dim=-1).cpu().numpy(),
        'embeds_t': text_embeds.cpu().numpy(),
        'embeds_i': image_embeds.cpu().numpy()
    }
    return ret
