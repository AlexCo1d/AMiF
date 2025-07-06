import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCAFusion(nn.Module):
    def __init__(self, hidden_size=768, n_heads=12):
        """
        Initializes the Gated Cross-Attention Fusion Module.

        Args:
            hidden_size (int): The dimensionality of the input and output features.
            n_heads (int): The number of attention heads.
        """
        super().__init__()

        # Multi-head cross-attention layer
        self.self_attention = cross_attention(feature_size=hidden_size, head=n_heads)
        self.cross_attention = cross_attention(feature_size=hidden_size, head=n_heads)
        # Gate mechanism: a linear layer followed by sigmoid activation
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        # self.norm3 = nn.LayerNorm(hidden_size)
        # self.norm4 = nn.LayerNorm(hidden_size)

        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.attn=None

    def forward(self, x, y):
        """
        Forward pass for the Gated Cross-Attention module.

        Args:
            x (Tensor): Fused features from earlier transformer layers.
                        Shape: (batch_size, seq_len_x, d_model)
            y (Tensor): Prior answers used as prior knowledge, word embedding.
                        Shape: (1, seq_len_y, d_model)
            x_mask (Tensor, optional): Mask for `x`. Shape: (batch_size, seq_len_x)
            y_mask (Tensor, optional): Mask for `y`. Shape: (batch_size, seq_len_y)

        Returns:
            Tensor: The output features after gated cross-attention.
        """
        # Cross-Attention
        # Query: x (fused features)
        # Key and Value: y (prior answers)

        y = self.self_attention(y, y, y)[0]
        y = self.ffn1(y) + y
        y = self.norm2(y)
        y = y.expand(x.size(0), -1, -1) # expand to bs
        x1, attn = self.cross_attention(x, y, y)
        self.attn=attn
        # Gate computation
        gate_values = self.gate(x1)  # Shape: (batch_size, seq_len_x, d_model)

        # Element-wise gated fusion of cross-attention output and original fused features
        gated_output = gate_values * x1 + (1 - gate_values) * x

        gated_output = self.ffn2(gated_output) + gated_output
        output = self.norm1(gated_output)

        return output


# code from M3AE/LaPA
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # bs,98,512 => 16,98,1
        std = x.std(-1, keepdim=True)  # # bs,98,512 => 16,98,1
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class cross_attention(nn.Module):
    def __init__(self, feature_size, head=8):
        super().__init__()
        self.cross_attn = MultiHeadedAttention(h=head, d_model=feature_size, dropout=0.1)
        self.layer_norm = LayerNorm(feature_size)
        self.attn=None

    def forward(self, q, k, v):
        t = self.layer_norm(q + self.cross_attn(q, k, v))
        return t, self.cross_attn.attn
