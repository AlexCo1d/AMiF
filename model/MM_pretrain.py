from typing import Any

import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision
from lightning.pytorch.utilities.types import STEP_OUTPUT
from open_clip import create_model_from_pretrained, get_tokenizer
from open_clip.tokenizer import HFTokenizer
from torch import distributed as dist
from einops import rearrange
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torchvision.transforms import InterpolationMode
from transformers import BertTokenizerFast, RobertaTokenizerFast, BertLMHeadModel, \
    AutoModel, AutoTokenizer, AutoProcessor
from transformers.models.bert.modeling_bert import BertConfig, BertModel

from Utils.misc import is_dist_avail_and_initialized, get_world_size
from model.submodule import local_conloss
from model.submodule.BLIP.BLIPBase import LayerNorm
from model.submodule.Loss_utils import KLContrastiveLoss, cost_matrix_cosine, ipot, trace, compute_precision_at_k
from model.submodule.bert.CrossLayer import BertCrossLayer
from model.submodule.vit.eva_vit import create_eva_vit_g
from model.submodule.vit.vit import get_ViT
from model.submodule.CLIP.vision_encoders.clip_model import build_model, adapt_position_encoding


class MM_pretrain(pl.LightningModule):
    def __init__(self,
                 config
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.max_txt_length = config['max_txt_length']
        image_size = config['image_size']
        # build for pre-training model
        if "clip" in self.hparams.config["vision_encoder"]:
            self.vision_encoder = build_model(config[config["vision_encoder"]]['vit'], resolution_after=image_size)
            self.ln_vision = nn.Identity()
        else:
            self.vision_encoder, self.ln_vision = self.init_vision_encoder(config[config["vision_encoder"]]['vit'],
                                                                           image_size)
        if 'roberta' in config['text_encoder']:
            from transformers import RobertaModel
            self.text_encoder = RobertaModel.from_pretrained(config['roberta']['text_encoder_path'])
            self.tokenizer = RobertaTokenizerFast.from_pretrained(config['roberta']['text_encoder_path'])
        elif 'bioclinicalbert' in config['text_encoder']:
            self.text_encoder = AutoModel.from_pretrained(config[config['text_encoder']]['text_encoder_path'])
            self.tokenizer = BertTokenizerFast.from_pretrained(config[config['text_encoder']]['text_encoder_path'])
        # else:
        #     self.text_encoder = AutoModel.from_pretrained(config[config['text_encoder']]['text_encoder_path'])
        #     self.tokenizer = BertTokenizerFast.from_pretrained(config[config['text_encoder']]['text_encoder_path'])

        if self.hparams.config[config['text_encoder']]['freeze'] is True or self.hparams.config['stage'] > 0:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        if self.hparams.config[config['vision_encoder']]['freeze'] is True or self.hparams.config['stage'] > 0:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        hidden_size = self.hparams.config['hidden_size']
        # local_proj_dim = self.hparams.config['local_proj_dim']
        # queue_size = self.hparams.config['queue_size']

        self.vision_cross_modal_proj = nn.Linear(hidden_size, hidden_size).apply(init_weights)  # 768, 256
        # self.vision_proj_local = nn.Linear(hidden_size, local_proj_dim).apply(init_weights)  # 768, 256
        self.text_cross_modal_proj = nn.Linear(hidden_size, hidden_size).apply(init_weights)
        # self.text_proj_local = nn.Linear(hidden_size, local_proj_dim).apply(init_weights)

        if config['early_fusion'] is True:
            self.modality_token_embeddings = nn.Embedding(2, hidden_size).apply(init_weights)
            bert_config = BertConfig.from_pretrained(config[config['text_encoder']]['text_encoder_path'])

            self.cross_modal_text_layers = nn.ModuleList(
                [BertCrossLayer(bert_config) for _ in range(config['cross_modal_layer'])])
            self.cross_modal_text_layers.apply(init_weights)
            self.cross_modal_image_layers = nn.ModuleList(
                [BertCrossLayer(bert_config) for _ in range(config['cross_modal_layer'])])
            self.cross_modal_image_layers.apply(init_weights)
            if self.hparams.config['itm']:
                self.itm_head = nn.Linear(2 * hidden_size, 2).apply(init_weights)

        self.softlabel = self.hparams.config['softlabel']
        self.multiview = self.hparams.config['multiview']
        self.uni_modal_text = self.hparams.config['uni_modal_text']
        self.IPOT = self.hparams.config['ipot']
        self.hard_negative = self.hparams.config['hard_negative']

        self.temp1 = nn.Parameter(self.hparams.config['temperature_1'] * torch.ones([]))
        self.temp2 = nn.Parameter(self.hparams.config['temperature_2'] * torch.ones([]))
        self.temp3 = nn.Parameter(self.hparams.config['temperature_3'] * torch.ones([]))
        # self.temp4 = nn.Parameter(self.hparams.config['temperature_4'] * torch.ones([]))
        # self.alpha = 0.4

        # for memory bank
        # self.register_buffer("image_queue", torch.randn(global_proj_dim, queue_size))
        # self.register_buffer("text_queue", torch.randn(global_proj_dim, queue_size))
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # self.queue_size = queue_size
        # self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        # self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        # validation list
        self.Images_feats = []
        self.Texts_feats = []
        self.Distances = []

        if self.softlabel:
            self.KD_model = AutoModel.from_pretrained(self.hparams.config['KD_model']['load_path'],
                                                      trust_remote_code=True)
            self.KD_tokenizer = AutoTokenizer.from_pretrained(self.hparams.config['KD_model']['load_path'])

            for name, param in self.KD_model.named_parameters():
                param.requires_grad = False

        # load pre-trained model's weights
        if self.hparams.config["load_path"][0] != "":
            for p in self.hparams.config["load_path"]:
                if p != "":
                    ckpt = torch.load(p, map_location="cpu")
                    if "state_dict" in ckpt:
                        state_dict = ckpt["state_dict"]
                    else:
                        state_dict = ckpt
                    self.load_state_dict(state_dict, strict=False)

    def forward(self, batch, batch_idx, mode='train'):
        if mode == 'train':
            return self.forward_train(batch, batch_idx)
        else:
            return self.forward_val(batch)

    def forward_train(self, batch, batch_idx):
        loss = {}

        # alpha = self.alpha * min(1, (
        #         self.current_epoch * len(self.trainer.train_dataloader) + batch_idx) / len(
        #     self.trainer.train_dataloader))

        image = batch["image1"]
        text = batch["text"]
        text_tokens = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt",
                                     max_length=self.max_txt_length).to(self.device)

        rank = dist.get_rank()
        bs = image.size(0)

        # forward encoders
        uni_modal_text_output = self.text_encoder(
            input_ids=text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True
        )
        uni_modal_text_feats = uni_modal_text_output.hidden_states[-1]

        uni_modal_image_feats = self.ln_vision(self.vision_encoder(image))

        # forward global features
        global_image_feats = F.normalize(
            self.vision_cross_modal_proj(uni_modal_image_feats[:, 0, :]), dim=-1
        )
        global_text_feats = F.normalize(
            self.text_cross_modal_proj(uni_modal_text_feats[:, 0, :]), dim=-1
        )

        # global_image_feats_q = torch.cat([global_image_feats.t(), self.image_queue.clone().detach()], dim=1)
        # global_text_feats_q = torch.cat([global_text_feats.t(), self.text_queue.clone().detach()], dim=1)

        global_image_feats_all = concat_all_gather(global_image_feats)  # [batch_size*num_gpu, embed_dim]
        global_text_feats_all = concat_all_gather(global_text_feats)  # [batch_size*num_gpu, embed_dim]

        # forward Multiview features
        if self.multiview:
            image2 = batch["image2"]
            if self.softlabel:
                _image2 = batch['_image2']

            uni_modal_image_feats2 = self.ln_vision(self.vision_encoder(image2))
            global_image_feats2 = F.normalize(
                self.vision_cross_modal_proj(uni_modal_image_feats2[:, 0, :]), dim=-1
            )
            global_image_feats_all2 = concat_all_gather(global_image_feats2)  # [batch_size*num_gpu, embed_dim]

        # generate softlabel
        if self.softlabel:
            _image1 = batch['_image1']

            self.KD_model.eval()
            KD_text_tokens = self.KD_tokenizer(text, padding='max_length', truncation=True, return_tensors="pt",
                                               max_length=self.max_txt_length).to(self.device)
            # print('!!!', KD_text_tokens)
            KD_output = self.KD_model(input_ids=KD_text_tokens.input_ids, attention_mask=KD_text_tokens.attention_mask,
                                      pixel_values=_image1, return_dict=True)
            logit_scale = self.KD_model.logit_scale.exp()
            KD_global_image_feats1 = KD_output.image_embeds
            KD_global_text_feats = KD_output.text_embeds
            KD_global_image_feats2 = self.KD_model.get_image_features(pixel_values=_image2, return_dict=True)

            KD_global_image_feats_all1 = concat_all_gather(KD_global_image_feats1)
            KD_global_image_feats_all2 = concat_all_gather(KD_global_image_feats2)
            KD_global_text_feats_all = concat_all_gather(KD_global_text_feats)

            # self._dequeue_and_enqueue(global_image_feats, global_text_feats)

            # forward global image-text Contrastive Loss
            # if softlabels are used, add mv and unimodal text (ut) to form the overall softlabel
            sim_i2t = global_image_feats @ global_text_feats_all.t()
            sim_t2i = global_text_feats @ global_image_feats_all.t()

            sim_targets = torch.zeros_like(sim_i2t).to(self.device)
            sim_targets.fill_diagonal_(1)

            sim_targets_i2t_soft = (logit_scale * KD_global_image_feats1 @ KD_global_text_feats_all.t()
                                    + sim_targets * self.hparams.config['coef_softlabel_target'])
            sim_targets_t2i_soft = (logit_scale * KD_global_text_feats @ KD_global_image_feats_all1.t()
                                    + sim_targets * self.hparams.config['coef_softlabel_target'])

            loss_itc = (
                               KLContrastiveLoss(sim_i2t, sim_targets_i2t_soft, self.temp1, 1.0, use_loss="kl")
                               + KLContrastiveLoss(sim_t2i, sim_targets_t2i_soft, self.temp1, 1.0, use_loss="kl")
                       ) / 2
        else:
            targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
                self.device
            )

            sim_i2t = global_image_feats @ global_text_feats_all.t() / self.temp1
            sim_t2i = global_text_feats @ global_image_feats_all.t() / self.temp1

            loss_itc = (
                               F.cross_entropy(sim_i2t, targets, label_smoothing=0.1, reduction='mean')
                               + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1, reduction='mean')
                       ) / 2

        loss['itgc'] = loss_itc

        if self.hard_negative is True:
            # given sim_i2t, select the hardest negative samples for each one.
            text_embeds_all = concat_all_gather(uni_modal_text_feats)
            text_embeds_neg = []
            for b in range(bs):
                mask = torch.ones(sim_i2t.shape[1], dtype=torch.bool).to(self.device)
                mask[b] = False
                neg_idx = torch.multinomial(sim_i2t[b][mask], 1).item()
                text_embeds_neg.append(text_embeds_all[neg_idx])
            text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
            uni_modal_text_feats = torch.cat([uni_modal_text_feats, text_embeds_neg], dim=0)
            uni_modal_image_feats = uni_modal_image_feats.repeat(2, 1, 1)

        # forward multiview & inter-text Contrastive Loss
        if self.multiview:
            sim_i1_i2 = global_image_feats @ global_image_feats_all2.t()
            sim_i2_i1 = global_image_feats2 @ global_image_feats_all.t()  # [batch_size, batch_size*num_gpu]

            targets_mv = torch.zeros_like(sim_i1_i2).to(self.device)
            targets_mv.fill_diagonal_(1)

            sim_targets_i1_i2_soft = (logit_scale * KD_global_image_feats1 @ KD_global_image_feats_all2.t() +
                                      targets_mv * self.hparams.config['coef_softlabel_target'])
            sim_targets_i2_i1_soft = (logit_scale * KD_global_image_feats2 @ KD_global_image_feats_all1.t() +
                                      targets_mv * self.hparams.config['coef_softlabel_target'])

            inter_view_loss = (
                                      KLContrastiveLoss(sim_i1_i2, sim_targets_i1_i2_soft, self.temp2)
                                      + KLContrastiveLoss(sim_i2_i1, sim_targets_i2_i1_soft, self.temp2)
                              ) / 2
            loss['imvc'] = inter_view_loss

        if self.uni_modal_text:
            sim_uni_text_label = global_text_feats @ global_text_feats_all.t()
            targets_ut = torch.zeros_like(sim_uni_text_label).to(self.device)
            targets_ut.fill_diagonal_(1)

            sim_targets_ut_soft = (logit_scale * KD_global_text_feats @ KD_global_text_feats_all.t() +
                                   targets_ut * self.hparams.config['coef_softlabel_target'])

            loss_ut = KLContrastiveLoss(sim_uni_text_label, sim_targets_ut_soft, self.temp3)
            loss['utgc'] = loss_ut

        # forward Local Loss
        if self.IPOT and self.hparams.config['stage'] > 0:
            if self.hparams.config['early_fusion']:
                image_masks = torch.ones((uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)),
                                         dtype=torch.long).to(
                    self.device)
                text_fused_feats, image_fused_feats, attentions = self.forward_cross_modal(image, uni_modal_text_feats,
                                                                                           uni_modal_image_feats,
                                                                                           text_tokens.attention_mask.repeat(
                                                                                               2,
                                                                                               1) if self.hard_negative else text_tokens.attention_mask,
                                                                                           image_masks)
                loss_wpa, _ = self.loss_wpa(image, image_fused_feats[:bs], text_tokens.input_ids,
                                            text_fused_feats[:bs], attentions,
                                            text_tokens.attention_mask,
                                            agg=self.hparams.config['aggregate_tokens'])
                loss['wpa'] = loss_wpa * self.hparams.config['wpa_weight']
            else:
                loss_wpa, _ = self.loss_wpa(image, uni_modal_image_feats, text_tokens.input_ids,
                                            uni_modal_text_output.hidden_states[-1],
                                            uni_modal_text_output.attentions[-1],
                                            text_tokens.attention_mask,
                                            agg=self.hparams.config['aggregate_tokens'])
                loss['wpa'] = loss_wpa * self.hparams.config['wpa_weight']
        # else:
        #     image_masks = torch.ones((uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)),
        #                              dtype=torch.long).to(
        #         self.device)
        #     text_fused_feats, image_fused_feats, attentions = self.forward_cross_modal(image, uni_modal_text_feats,
        #                                                                                uni_modal_image_feats,
        #                                                                                text_tokens.attention_mask,
        #                                                                                image_masks)
        #     loss_wpa, _ = self.forward_local_contrastive_loss(image_fused_feats, text_tokens.input_ids,
        #                                                       text_fused_feats, attentions)
        #     loss['local'] = loss_wpa * self.hparams.config['wpa_weight']

        if self.hparams.config['early_fusion'] and self.hparams.config['itm'] and self.hparams.config[
            'stage'] > 0:
            cls_feats = torch.cat([text_fused_feats[:, 0, :], image_fused_feats[:, 0, :]], dim=-1)
            itm_logits = self.itm_head(cls_feats)
            itm_logits = torch.clamp(itm_logits, min=1e-7, max=1 - 1e-7)  # avoid nan
            if self.hard_negative is True:
                itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(bs, dtype=torch.long)]).to(
                    self.device)
            else:
                itm_labels = torch.zeros(bs, dtype=torch.long).to(self.device)
            loss_itm = F.cross_entropy(itm_logits, itm_labels)
            loss['itm'] = loss_itm
        return loss

    def forward_cross_modal(self, image, uni_modal_text_feats, uni_modal_image_feats, text_masks, image_masks):
        global_image_feats = self.vision_cross_modal_proj(uni_modal_image_feats)
        uni_modal_text_feats = self.text_cross_modal_proj(uni_modal_text_feats)
        extended_image_masks = self.text_encoder.get_extended_attention_mask(image_masks,
                                                                             image_masks.size()).to(
            self.device)
        extended_text_masks = self.text_encoder.get_extended_attention_mask(text_masks,
                                                                            text_masks.size()).to(
            self.device)

        # apply token embeddings for modality
        uni_modal_text_feats, uni_modal_image_feats = (
            uni_modal_text_feats + self.modality_token_embeddings(
                torch.zeros_like(text_masks)),
            global_image_feats + self.modality_token_embeddings(torch.full_like(image_masks, 1)),
        )

        # Co-attention layer
        x, y = uni_modal_text_feats, uni_modal_image_feats
        for layer_idx, (text_layer, image_layer) in enumerate(zip(self.cross_modal_text_layers,
                                                                  self.cross_modal_image_layers)):
            x1 = text_layer(x, y, extended_text_masks, extended_image_masks, output_attentions=True)
            y1 = image_layer(y, x, extended_image_masks, extended_text_masks, output_attentions=True)
            x, y = x1[0], y1[0]
        return x, y, (x1[1:], y1[1:])

    def forward_val(self, batch):
        image = batch["image"]
        text = batch["text"]
        text_tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt",
                                     max_length=self.max_txt_length).to(image.device)

        # rank = dist.get_rank()
        # bs = image.size(0)
        uni_modal_text_output = self.text_encoder(
            input_ids=text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True
        )
        uni_modal_text_feats = uni_modal_text_output.hidden_states[-1]

        uni_modal_image_feats = self.ln_vision(self.vision_encoder(image))

        # forward global features
        global_image_feats = F.normalize(
            self.vision_cross_modal_proj(uni_modal_image_feats[:, 0, :]), dim=-1
        )
        global_text_feats = F.normalize(
            self.text_cross_modal_proj(uni_modal_text_feats[:, 0, :]), dim=-1
        )
        image_masks = torch.ones((uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)), dtype=torch.long).to(
            self.device)
        Output = None
        if self.hparams.config['early_fusion'] and self.hparams.config['stage'] > 0:
            text_fused_feats, image_fused_feats, attentions = self.forward_cross_modal(image, uni_modal_text_feats,
                                                                                       uni_modal_image_feats,
                                                                                       text_tokens.attention_mask,
                                                                                       image_masks)
            _, Output = self.loss_wpa(image, image_fused_feats, text_tokens.input_ids, text_fused_feats,
                                      attentions,
                                      text_tokens.attention_mask,
                                      beta=0.1, iteration=1000, k=1,
                                      agg=self.hparams.config['aggregate_tokens'])
        # else:
        #     # local sim
        #     _, Output = self.loss_wpa(image, uni_modal_image_feats, text_tokens.input_ids,
        #                               uni_modal_text_output.hidden_states[-1],
        #                               uni_modal_text_output.attentions[-1],
        #                               text_tokens.attention_mask,
        #                               beta=0.1, iteration=1000, k=1,
        #                               agg=self.hparams.config['aggregate_tokens'])
        if not self.IPOT:
            text_fused_feats, image_fused_feats, attentions = self.forward_cross_modal(image, uni_modal_text_feats,
                                                                                       uni_modal_image_feats,
                                                                                       text_tokens.attention_mask,
                                                                                       image_masks)
            _, Output = self.forward_local_contrastive_loss(image_fused_feats, text_tokens.input_ids,
                                                            text_fused_feats, attentions)

        return (global_image_feats, global_text_feats, Output)

    def validation_step(self, batch, batch_idx):
        o = self(batch, batch_idx, mode='val')
        global_image_feats, global_text_feats = o[0], o[1]
        global_image_feats = concat_all_gather(global_image_feats)
        global_text_feats = concat_all_gather(global_text_feats)
        self.Images_feats.append(global_image_feats.cpu())
        self.Texts_feats.append(global_text_feats.cpu())

        return global_image_feats, global_text_feats

    def on_validation_epoch_end(self):
        global_image_feats = torch.cat(self.Images_feats, dim=0).cpu()
        global_text_feats = torch.cat(self.Texts_feats, dim=0).cpu()

        scores_i2t = global_image_feats @ global_text_feats.t()
        scores_t2i = global_text_feats @ global_image_feats.t()
        # print(scores_i2t.shape, scores_t2i.shape)
        scores_i2t = F.softmax(scores_i2t, dim=-1).numpy()
        scores_t2i = F.softmax(scores_t2i, dim=-1).numpy()

        classes = np.repeat(np.arange(5), 200)
        eval_result = {
            "i2t_r1": compute_precision_at_k(scores_i2t, classes, k=1),
            # "i2t_r2": compute_precision_at_k(scores_i2t, classes, k=2),
            "i2t_r5": compute_precision_at_k(scores_i2t, classes, k=5),
            "i2t_r10": compute_precision_at_k(scores_i2t, classes, k=10),
            # "i2t_r50": compute_precision_at_k(scores_i2t, classes, k=50),
            "t2i_r1": compute_precision_at_k(scores_t2i, classes, k=1),
            # "t2i_r2": compute_precision_at_k(scores_t2i, classes, k=2),
            "t2i_r5": compute_precision_at_k(scores_t2i, classes, k=5),
            "t2i_r10": compute_precision_at_k(scores_t2i, classes, k=10),
            # "t2i_r50": compute_precision_at_k(scores_t2i, classes, k=50),
        }
        val_metric = (eval_result['i2t_r1'] + eval_result['t2i_r1'] + eval_result['i2t_r5'] + eval_result['t2i_r5']) / 4
        eval_result.update({'val_metric': val_metric})
        # print(eval_result)
        self.log_dict(eval_result, batch_size=self.hparams.config['valid']['val_batch_size'], sync_dist=True,
                      prog_bar=True, logger=True, on_step=False, on_epoch=True)

        del self.Images_feats[:], self.Texts_feats[:], self.Distances[:]

    def on_validation_epoch_start(self):
        # clear the list first
        del self.Images_feats[:], self.Texts_feats[:], self.Distances[:]

    def training_step(self, batch, batch_idx):
        loss = self(batch, batch_idx, mode='train')
        loss_values = []
        for i in loss.keys():
            loss_values.append(loss[i])
        loss_sum = torch.sum(torch.stack(loss_values), dim=0)

        self.log_dict(loss, batch_size=self.hparams.config['batch_size'],
                      sync_dist=True, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss_sum

    def configure_optimizers(self):
        # max_steps = (
        #         len(self.trainer.train_dataloader)
        #         * self.hparams.config['max_epochs']
        #         // (self.hparams.config['accumulate_grad_steps'] * max(1, len(self.hparams.config['device'])))
        # )
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.config['lr'],
            betas=(0.9, 0.98),
            weight_decay=self.hparams.config['weight_decay']
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.config['lr'],
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.4)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def loss_wpa(self, image, image_embeds, input_ids, text_embeds, attentions,
                 text_attention_mask, beta=0.5, iteration=50, k=1, agg=False):
        B = image.size(0)
        # generate random negative pair
        # pos_len = B // 2
        # neg_len = B - pos_len
        # itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        #     self.device
        # )
        # itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]
        #
        # mixed_images_embeds = torch.stack([
        #     ti if itm_labels[i] == 1 else fi
        #     for i, (ti, fi) in enumerate(zip(image, false_image))
        # ])  # [B, (images_dim)]
        # mixed_images_embeds = self.ln_vision(self.vision_encoder(mixed_images_embeds))  # [B, len, dim]

        images_embeds = F.normalize(
            image_embeds, dim=-1
        )
        # remove the cls token for image
        images_embeds = images_embeds[:, 1:].contiguous()
        image_mask = torch.ones(images_embeds.size()[:-1]).bool().to(self.device)

        if agg:
            # aggregate the text tokens
            if self.hparams.config['early_fusion']:
                text_embeds, sents, last_atten_pt = local_conloss.aggregate_tokens(self,
                                                                                   text_embeds.unsqueeze(1),
                                                                                   input_ids,
                                                                                   attentions[0][0][:, :, 0, 1:].mean(
                                                                                       dim=1))
            else:
                text_embeds, sents, last_atten_pt = local_conloss.aggregate_tokens(self,
                                                                                   text_embeds.unsqueeze(
                                                                                       1),
                                                                                   input_ids,
                                                                                   attentions[:, :, 0, 1:].mean(dim=1))
            text_mask = last_atten_pt[:, 1:] != 0
            # remove the cls token for text
            text_embeds = F.normalize(
                text_embeds, dim=-1
            )
            text_embeds = text_embeds[:, 0]
            text_embeds = text_embeds[:, 1:].contiguous()
            sents = sents[0][1:-1]  # for vis, so bs=1
        else:
            # no aggregate
            text_embeds = F.normalize(
                text_embeds, dim=-1
            )
            # remove the cls token for text
            text_embeds = text_embeds[:, 1:]
            text_mask = text_attention_mask[:, 1:].bool().to(self.device)
            sents = self.tokenizer.convert_ids_to_tokens(input_ids[0])[
                    1:-1]  # only for the visualization use, remove the [CLS] and [SEP] token, so bs=1

        # no calculate the SEP token
        for i, _len in enumerate(text_mask.sum(dim=1)):
            text_mask[i, _len - 1] = False

        txt_pad, img_pad = ~text_mask, ~image_mask

        cost = cost_matrix_cosine(text_embeds.float(), images_embeds.float())
        joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
        cost.masked_fill_(joint_pad, 0)

        txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        T = ipot(
            cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta=beta, iteration=iteration, k=k
        )
        distance = trace(cost.matmul(T.detach()))
        # dist_pos = distance.masked_select(itm_labels == 1)
        # dist_neg = distance.masked_select(itm_labels == 0)
        # ot_loss = (dist_pos.sum() - dist_neg.sum()) / (dist_pos.size(0) + dist_neg.size(0))
        ot_loss = distance.mean()
        return ot_loss, {'T': cost.detach().cpu(), 'sents': sents}

    def init_vision_encoder(self, vit_path, img_size, drop_path_rate=0, use_grad_checkpoint=False, precision='fp16'
                            , encoder='vit16'):
        if encoder != 'eva_vit':
            print('load VIT model')
            model = get_ViT(vit_path, img_size, drop_path_rate=drop_path_rate)
            return model, nn.Identity()
        else:
            visual_encoder = create_eva_vit_g(vit_path, img_size, drop_path_rate, use_grad_checkpoint, precision)
            ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    def forward_local_contrastive_loss(self, img_features, ids, words_emb, attentions):
        """
        :param ids: caption_ids from tokenizer
        :param img_features: [b, patch_num==query_num, v_embed]
        :param words_emb: bert output
        :param attentions: text_attention for last layer
        :return: loss
        """

        temperature = 0.07
        # get the local word embed
        bz = img_features.size(0)
        all_feat = words_emb.unsqueeze(1)  # [b, layer, words_length, embed]
        last_layer_attn = attentions[0][0][:, :, 0, 1:].mean(dim=1)

        # t = time.time()
        all_feat, sents, word_atten = local_conloss.aggregate_tokens(self, all_feat,
                                                                     ids, last_layer_attn)
        # print("time for aggregate_tokens", time.time() - t)
        word_atten = word_atten[:, 1:].contiguous()
        all_feat = all_feat[:, 0]
        # report_feat = all_feat[:, 0].contiguous()
        word_feat = all_feat[:, 1:].contiguous()  # [b, words_length, embed]
        # we get report_feat, word_feat, last_atten_pt, sents now
        # word_emb = self.text_local_embedding(word_feat)
        word_emb = F.normalize(word_feat, dim=-1)
        # words_emb: [b, words_length, embed_dim=768]

        # same to the image features because they are all transformer based
        # img_feat=img_features[-1, :, 0].contiguous()  # [b, embed]
        patch_feat = img_features[:, 1:].contiguous()  # [b, patch_num, v_embed]

        # img_features = img_features.sum(axis=1)  # [b, patch_num, embed]
        # img_features = img_features.permute(0, 2, 1)
        # img_features = img_features / torch.norm(
        #     img_features, 2, dim=1, keepdim=True
        # ).expand_as(img_features)

        # we get img_feat and patch_feat now
        # patch_emb = self.vision_local_embedding(patch_feat)
        patch_emb = F.normalize(patch_feat, dim=-1)  # [b, patch_num, embed=768]

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
            img_attn_map = attentions[1][0].detach()
            atten_weights = img_attn_map[:, :, 0, :].mean(dim=1)
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

        return loss_local, {'T': atten_sim.detach().cpu(), 'sents': sents}

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_devices)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices
        print('total_training_steps', num_training_steps := (dataset_size // effective_batch_size) * trainer.max_epochs)
        return num_training_steps


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


def init_weights(self):
    if isinstance(self, (nn.Linear, nn.Embedding)):
        self.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(self, nn.LayerNorm):
        self.bias.data.zero_()
        self.weight.data.fill_(1.0)

    if isinstance(self, nn.Linear) and self.bias is not None:
        self.bias.data.zero_()
