import copy
import os.path

import numpy as np
from transformers import BertTokenizer
import torch.distributed as dist

from model.archi_mvcm import concat_all_gather
from model.submodule.BLIP.BLIPBase import (Blip2Base, disabled_train)

"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import string
from packaging import version
import torch.nn.functional as F
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from model.submodule.bert.xbert import BertConfig, BertModel, BertLMHeadModel


class Former_Clsvqa(Blip2Base):

    def __init__(
            self,
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=False,
            num_query_token=32,
            prompt="",
            max_txt_len=256,
            max_output_txt_len=256,
            vit_type="eva_vit",
            vit_path="",
            bert='bert-base-uncased',
            qformer_text_input=True,
            instruct=True,
            distill=False
    ):
        super().__init__()
        from transformers import LlamaTokenizer
        from model.submodule.LLM.modeling_llama import LlamaForCausalLM
        if not bert.startswith('.'):
            tokenizer_config = os.path.join('../model/submodule/bert/', bert)
        else:
            tokenizer_config = bert

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_config)
        self.tokenizer.add_special_tokens({"eos_token": "[SEP]"})
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(vit_path, img_size, drop_path_rate,
                                                                       use_grad_checkpoint, vit_precision,
                                                                       encoder=vit_type)
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, tokenizer_config=tokenizer_config
        )

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        self.qformer_text_input = qformer_text_input
        self.max_txt_len = max_txt_len
        self.instruct = instruct
        self.distill = distill

        config = BertConfig.from_json_file(os.path.join(tokenizer_config, 'config.json'))
        # config.fusion_layer = 0
        # config.num_hidden_layers = 6
        self.text_decoder = BertLMHeadModel.from_pretrained(tokenizer_config, config=config)

        if self.distill:
            self.Qformer_m = copy.deepcopy(self.Qformer)
            self.text_decoder_m = copy.deepcopy(self.text_decoder)
            self.model_pairs = [[self.Qformer, self.Qformer_m],
                                [self.text_decoder, self.text_decoder_m]]
            self.copy_params()
            self.momentum = 0.995
            self.alpha = 0.4

    def forward(self, samples, dataloader=None, alpha=None):
        image = samples["image"].to(self.device)
        if alpha is not None:
            self.alpha = alpha

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        bs = image.size(0)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        text_Qformer = self.tokenizer(
            samples["text_input"],
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        answer = self.tokenizer(
            [i+self.tokenizer.eos_token for i in samples['text_output']],
            padding='longest',
            return_tensors="pt"
        ).to(image.device)
        answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        query_output = self.Qformer.bert(
            text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        query_atts = torch.ones(query_output.last_hidden_state.size()[:-1], dtype=torch.long).to(image.device)
        if self.distill:
            self._momentum_update()
            with torch.no_grad():
                query_output_m = self.Qformer_m.bert(text_Qformer.input_ids,
                                                     attention_mask=Qformer_atts,
                                                     query_embeds=query_tokens,
                                                     encoder_hidden_states=image_embeds,
                                                     encoder_attention_mask=image_atts,
                                                     return_dict=True)
                query_atts_m = torch.ones(query_output_m.last_hidden_state.size()[:-1], dtype=torch.long).to(
                    image.device)
                logits_m = self.text_decoder_m(answer.input_ids,
                                               attention_mask=answer.attention_mask,
                                               encoder_hidden_states=query_output_m.last_hidden_state,
                                               encoder_attention_mask=query_atts_m,
                                               return_logits=True,
                                               )

            answer_output = self.text_decoder(answer.input_ids,
                                              attention_mask=answer.attention_mask,
                                              encoder_hidden_states=query_output.last_hidden_state,
                                              encoder_attention_mask=query_atts,
                                              labels=answer_targets,
                                              return_dict=True,
                                              soft_labels=F.softmax(logits_m, dim=-1),
                                              alpha=self.alpha,
                                              reduction='none',
                                              )
        else:
            answer_output = self.text_decoder(answer.input_ids,
                                              attention_mask=answer.attention_mask,
                                              encoder_hidden_states=query_output.last_hidden_state,
                                              encoder_attention_mask=query_atts,
                                              labels=answer_targets,
                                              return_dict=True,
                                              reduction='none',
                                              )

        return answer_output.loss.sum() / bs

    @torch.no_grad()
    def predict_answers(
            self,
            samples,
            dataloader,
            k=128
    ):
        image = samples["image"].to(self.device)
        bs = image.size(0)
        answer_list = dataloader.dataset.answer_list
        k = min(k, len(answer_list))
        answer_tokens = self.tokenizer(answer_list, padding='longest', return_tensors="pt").to(image.device)
        query_tokens = self.query_tokens.expand(bs, -1, -1)
        text_Qformer = self.tokenizer(
            samples["text_input"],
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        image = image.half()
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_output = self.Qformer.bert(
            text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        num_ques = query_output.last_hidden_state.size(0)  # num_ques = batch_size_test
        answer_ids = answer_tokens.input_ids
        answer_atts = answer_tokens.attention_mask
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        query_atts = torch.ones(query_output.last_hidden_state.size()[:-1], dtype=torch.long).to(image.device)
        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=query_output.last_hidden_state,
                                         encoder_attention_mask=query_atts,
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

        query_output.last_hidden_state = tile(query_output.last_hidden_state, 0, k)
        query_atts = tile(query_atts, 0, k)
        # torch.cuda.empty_cache()
        output = self.text_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=query_output.last_hidden_state,
                                   encoder_attention_mask=query_atts,
                                   labels=targets_ids,
                                   return_dict=True,
                                   reduction='none')

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)
        topk_probs = F.softmax(log_probs_sum, dim=-1)

        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)
        result = []
        for topk_id, topk_prob in zip(topk_ids, topk_probs):
            _, pred = topk_prob.max(dim=0)
            result.append(answer_list[topk_id[pred]])
        return result
        #
        # log_probs_sum = -output.loss
        # log_probs_sum = log_probs_sum.view(num_ques, k)
        #
        # max_topk_ids = log_probs_sum.argmax(dim=1)
        # max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]
        #
        # answers = [answer_list[max_id] for max_id in max_ids]
        #
        # return answers

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


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))
