import contextlib
import logging
import os.path

import torch
from torch import nn

from model.submodule.BLIP import QFormer
from model.submodule.vit.eva_vit import create_eva_vit_g
from model.submodule.vit.vit import get_ViT


class Blip2Base(nn.Module):

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2,
                     tokenizer_config='./model/submodule/bert/bert-base-uncased', checkpoint=True):
        encoder_config = QFormer.BertConfig.from_pretrained(tokenizer_config)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = QFormer.BertLMHeadModel.from_pretrained(tokenizer_config, config=encoder_config)
        # Qformer = QFormer.BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        return Qformer, query_tokens


    def init_vision_encoder(self, vit_path, img_size, drop_path_rate, use_grad_checkpoint, precision, encoder='eva_vit'):
        if encoder!='eva_vit':
            print('load VIT model')
            model = get_ViT(vit_path, img_size, drop_path_rate=drop_path_rate)
            return model, nn.Identity()
        else:
            visual_encoder= create_eva_vit_g(vit_path, img_size, drop_path_rate, use_grad_checkpoint, precision)
            ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def device(self):
        return list(self.parameters())[0].device

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)