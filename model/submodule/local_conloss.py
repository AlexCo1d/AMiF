import torch
from torch import nn


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x.permute(0, 2, 1)


def aggregate_tokens(self, embeddings, caption_ids, last_layer_attn):
    """
    :param embeddings: bz, layer, num_words, 768
    :param caption_ids: bz, 112
    :param last_layer_attn: bz, 111
    """
    _, num_layers, num_words, dim = embeddings.shape
    embeddings = embeddings.permute(0, 2, 1, 3)
    agg_embs_batch = []
    sentences = []
    last_attns = []
    # loop over batch
    for embs, caption_id, last_attn in zip(embeddings, caption_ids, last_layer_attn):
        agg_embs = []
        token_bank = []
        words = []
        word_bank = []
        attns = []
        attn_bank = []
        caption_id = self.tokenizer.convert_ids_to_tokens(caption_id)
        # loop over sentence
        for word_emb, word, attn in zip(embs, caption_id, last_attn):
            if word == "[SEP]":
                new_emb = torch.stack(token_bank)
                new_emb = new_emb.sum(axis=0)
                agg_embs.append(new_emb)
                words.append("".join(word_bank))
                attns.append(sum(attn_bank))
                agg_embs.append(word_emb)
                words.append(word)
                attns.append(attn)
                break
            # This is because some words are divided into two words.
            if not word.startswith("##"):
                if len(word_bank) == 0:
                    token_bank.append(word_emb)
                    word_bank.append(word)
                    attn_bank.append(attn)
                else:
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    attns.append(sum(attn_bank))

                    token_bank = [word_emb]
                    word_bank = [word]
                    attn_bank = [attn]
            else:
                token_bank.append(word_emb)
                word_bank.append(word[2:])
                attn_bank.append(attn)
        agg_embs = torch.stack(agg_embs)
        padding_size = num_words - len(agg_embs)
        paddings = torch.zeros(padding_size, num_layers, dim)
        paddings = paddings.type_as(agg_embs)
        words = words + ["[PAD]"] * padding_size
        last_attns.append(
            torch.cat([torch.tensor(attns), torch.zeros(padding_size)], dim=0))
        agg_embs_batch.append(torch.cat([agg_embs, paddings]))
        sentences.append(words)

    agg_embs_batch = torch.stack(agg_embs_batch)
    agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
    last_atten_pt = torch.stack(last_attns)
    last_atten_pt = last_atten_pt.type_as(agg_embs_batch)
    return agg_embs_batch, sentences, last_atten_pt
