import time
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch import distributed as dist
from transformers import RobertaModel, BertTokenizerFast, RobertaTokenizerFast, AutoModel, AutoTokenizer
from transformers.models.bert.modeling_bert import BertConfig

from VQA.vqaTools.vqaEvaluate import runtime_vqa_acc
from model.MM_pretrain import concat_all_gather
from model.submodule.Gated_Cross_Attention import GCAFusion
from model.submodule.bert.CrossLayer import BertCrossLayer
from model.submodule.bert.xbert import BertLMHeadModel

from model.submodule.CLIP.vision_encoders.clip_model import build_model, adapt_position_encoding


class MM_VQA(pl.LightningModule):
    def __init__(self,
                 config,
                 dataloader=None
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
            self.visual_encoder, self.ln_vision = self.init_vision_encoder(config[config["vision_encoder"]]['vit'],
                                                                           image_size)
        if 'roberta' in config['text_encoder']:
            self.text_encoder = RobertaModel.from_pretrained(config['roberta']['text_encoder_path'])
            self.tokenizer = RobertaTokenizerFast.from_pretrained(config['roberta']['text_encoder_path'])
        elif 'bioclinicalbert' in config['text_encoder']:
            self.text_encoder = AutoModel.from_pretrained(config[config['text_encoder']]['text_encoder_path'])
            self.tokenizer = BertTokenizerFast.from_pretrained(config[config['text_encoder']]['text_encoder_path'])

        self.tokenizer.add_special_tokens({"eos_token": "[SEP]"})
        # self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        # else:
        #     self.text_encoder = AutoModel.from_pretrained(config[config['text_encoder']]['text_encoder_path'])
        #     self.tokenizer = BertTokenizerFast.from_pretrained(config[config['text_encoder']]['text_encoder_path'])

        if self.hparams.config[config['text_encoder']]['freeze'] is True:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # for modality fusion
        bert_config = BertConfig.from_pretrained(config[config['text_encoder']]['text_encoder_path'])
        hidden_size = self.hparams.config['hidden_size']

        self.text_cross_modal_proj = nn.Linear(hidden_size, hidden_size).apply(init_weights)
        self.vision_cross_modal_proj = nn.Linear(hidden_size, hidden_size).apply(init_weights)
        self.modality_token_embeddings = nn.Embedding(2, hidden_size).apply(init_weights)

        self.cross_modal_text_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config['cross_modal_layer'])])
        self.cross_modal_text_layers.apply(init_weights)
        self.cross_modal_image_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config['cross_modal_layer'])])
        self.cross_modal_image_layers.apply(init_weights)

        # for text decoder
        self.text_decoder = BertLMHeadModel.from_pretrained(config[config['text_decoder']]['text_encoder_path'])

        self._total_answer = dataloader.dataset.total_answers
        self.prior_knowledge_tokens = None
        self.prior_knowledge_embedding = None
        self.gated_cross_attention = GCAFusion(hidden_size=hidden_size)

        # for validation metrics
        self.total_answer = []
        self.total_predict = []

        # load pre-trained model's weights
        if self.hparams.config["load_path"][0] != "":
            for p in self.hparams.config["load_path"]:
                ckpt = torch.load(p, map_location="cpu")
                if "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                else:
                    state_dict = ckpt
                self.load_state_dict(state_dict, strict=False)

    def forward(self, batch, batch_idx, mode='train'):
        image = batch["image"]
        text_input = batch["text_input"]
        text_output = batch["text_output"]
        image = image.cuda()
        text_input_tokens = self.tokenizer(text_input, padding='max_length', truncation=True, return_tensors="pt",
                                           max_length=self.max_txt_length).to(self.device)
        text_output_tokens = self.tokenizer(text_output,
                                            padding='longest', truncation=True, return_tensors="pt",
                                            max_length=self.max_txt_length).to(self.device)
        # rank = dist.get_rank()
        bs = image.size(0)
        if self.prior_knowledge_tokens is None:
            self.prior_knowledge_tokens = self.tokenizer(self._total_answer, padding='longest', max_length=50000,
                                                         return_tensors="pt").to(self.device)
        sents=self.tokenizer.convert_ids_to_tokens(self.prior_knowledge_tokens.input_ids[0])
        # get prior knowledge embedding
        prior_knowledge_embedding = self.text_encoder.embeddings.word_embeddings(
            self.prior_knowledge_tokens.input_ids
        )  # bs_prior=1

        # forward encoders
        uni_modal_question_output = self.text_encoder(
            input_ids=text_input_tokens.input_ids,
            attention_mask=text_input_tokens.attention_mask,
            return_dict=True
        )
        uni_modal_question_feats = uni_modal_question_output.hidden_states[-1]

        uni_modal_image_feats = self.ln_vision(self.vision_encoder(image))

        # forward cross-modal
        image_masks = torch.ones((uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)), dtype=torch.long).to(
            image.device)
        x, y = self.forward_cross_modal(image, uni_modal_question_feats, uni_modal_image_feats,
                                        text_input_tokens.attention_mask, image_masks)
        fusion_feats = torch.cat([x, y], dim=1)
        fusion_feats_attention = torch.cat([text_input_tokens.attention_mask, image_masks],
                                           dim=1)  #  fusion_feats_attention = torch.ones(fusion_feats.size(0), fusion_feats.size(1)).to(self.device)

        if self.hparams.config['use_gated_cross_attention']:
            fusion_feats = self.gated_cross_attention(fusion_feats, prior_knowledge_embedding)
            # fusion_feats_attention = torch.ones_like(fusion_feats_attention)

        # forward train
        if mode == 'train':
            # text_output_tokens.input_ids[:, 0] = self.tokenizer.bos_token_id  # signal the generating.
            answer_target = text_output_tokens.input_ids.masked_fill(
                text_output_tokens.input_ids == self.tokenizer.pad_token_id, -100)
            # forward decoders
            text_output_output = self.text_decoder(
                input_ids=text_output_tokens.input_ids,
                attention_mask=text_output_tokens.attention_mask,
                encoder_hidden_states=fusion_feats,
                encoder_attention_mask=fusion_feats_attention,
                # output_attentions=True,
                labels=answer_target,
                return_dict=True,
                reduction='none',
            )

            return text_output_output.loss.sum() / bs, (self.gated_cross_attention.attn, sents)

        # forward val
        elif mode == 'val':
            k = 32
            answer_list = self.trainer.val_dataloaders.dataset.answer_list
            answer_list_tokens = self.tokenizer(answer_list, padding='longest', max_length=self.max_txt_length,
                                                return_tensors="pt").to(self.device)

            # answer_list_tokens.input_ids[:, 0] = self.tokenizer.bos_token_id
            answer_ids = answer_list_tokens.input_ids

            answer_atts = answer_list_tokens.attention_mask
            start_ids = answer_ids[0, 0].repeat(bs, 1)
            start_output = self.text_decoder(start_ids,
                                             encoder_hidden_states=fusion_feats,
                                             encoder_attention_mask=fusion_feats_attention,
                                             return_dict=True,
                                             reduction='none')
            logits = start_output.logits[:, 0, :]

            answer_first_token = answer_ids[:, 1]
            prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)

            topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

            input_ids = []
            input_atts = []
            for b, topk_id in enumerate(topk_ids):
                input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
                input_atts.append(answer_atts.index_select(dim=0, index=topk_id))

            input_ids = torch.cat(input_ids, dim=0)
            input_atts = torch.cat(input_atts, dim=0)

            targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

            fusion_feats = tile(fusion_feats, 0, k)
            fusion_feats_attention = tile(fusion_feats_attention, 0, k)
            output = self.text_decoder(input_ids,
                                       attention_mask=input_atts,
                                       encoder_hidden_states=fusion_feats,
                                       encoder_attention_mask=fusion_feats_attention,
                                       labels=targets_ids,
                                       return_dict=True,
                                       reduction='none')

            answer_loss = output.loss
            answer_loss = answer_loss.view(input_ids.size(0), -1)

            # topk_prob: first token probability
            topk_probs = topk_probs.view(-1, 1)
            log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

            log_probs_sum = log_probs.sum(1)
            log_probs_sum = log_probs_sum.view(bs, k)
            topk_probs = F.softmax(log_probs_sum, dim=-1)

            # get top-k after re-ranking
            topk_probs, rerank_id = topk_probs.topk(k, dim=1)
            topk_ids = torch.gather(topk_ids, 1, rerank_id)
            result = []
            for topk_id, topk_prob in zip(topk_ids, topk_probs):
                _, pred = topk_prob.max(dim=0)
                result.append(answer_list[topk_id[pred]])

            return result

    def forward_cross_modal(self, image, uni_modal_text_feats, uni_modal_image_feats, text_masks, image_masks):
        global_image_feats = self.vision_cross_modal_proj(uni_modal_image_feats)
        uni_modal_text_feats = self.text_cross_modal_proj(uni_modal_text_feats)

        extended_image_masks = self.text_encoder.get_extended_attention_mask(image_masks,
                                                                             image_masks.size()).to(
            image.device)
        extended_text_masks = self.text_encoder.get_extended_attention_mask(text_masks,
                                                                            text_masks.size()).to(
            image.device)

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

        return x, y

    def training_step(self, batch, batch_idx):
        loss,_ = self(batch, batch_idx, mode='train')
        loss = {'loss': loss}
        self.log_dict(loss, batch_size=self.hparams.config['batch_size'],
                      sync_dist=True, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        results = self(batch, batch_idx, mode='val')
        self.total_predict.extend(results)
        self.total_answer.extend(batch['text_output'])
        return results

    def on_validation_epoch_end(self):
        t = time.time()-self.time
        results = runtime_vqa_acc(self.total_answer, self.total_predict)
        results.update({'time':t})
        self.log_dict(results, batch_size=self.hparams.config['valid']['val_batch_size'], sync_dist=True,
                      prog_bar=True, logger=True, on_step=False, on_epoch=True)
        del self.total_answer[:], self.total_predict[:]

    def on_validation_epoch_start(self):
        self.time = time.time()
        del self.total_answer[:], self.total_predict[:]

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

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_devices)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices
        print('total_training_steps', num_training_steps := (dataset_size // effective_batch_size) * trainer.max_epochs)
        return num_training_steps


def init_weights(self):
    if isinstance(self, (nn.Linear, nn.Embedding)):
        self.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(self, nn.LayerNorm):
        self.bias.data.zero_()
        self.weight.data.fill_(1.0)

    if isinstance(self, nn.Linear) and self.bias is not None:
        self.bias.data.zero_()


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))
