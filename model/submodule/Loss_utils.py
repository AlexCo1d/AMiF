import numpy as np
import torch
import torch.nn.functional as F


def KLContrastiveLoss(logits, softlabel, temp, softlabel_temp=1.0, reduction='mean', use_loss="kl"):
    """
    KL divergence loss
    make logits and softlabel have the same distribution
    logits to softlabel
    """
    # softmax for softlabel
    sim_targets = F.softmax(softlabel / softlabel_temp, dim=1)

    # log softmax
    logit_inputs = F.log_softmax(logits / temp, dim=1)

    if use_loss == "kl":
        # KL divergence
        loss = F.kl_div(logit_inputs, sim_targets, reduction='batchmean')
    elif use_loss == "contrastive":
        # Switch to the same loss as ContrastiveLoss, but sim_targets is soft
        if reduction == "mean":
            loss = -torch.mean(logit_inputs * sim_targets, dim=1).mean()
        elif reduction == "sum":
            loss = -torch.sum(logit_inputs * sim_targets, dim=1).mean()
        else:
            raise ValueError("contrastive loss mode error")
    else:
        raise ValueError("loss mode error")

    return loss


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(
        txt_emb, img_emb, txt_pad, img_pad, beta=0.5, iteration=50, k=1
):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(
        cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta, iteration, k
    )
    distance = trace(cost.matmul(T.detach()))
    return distance


def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def compute_precision_at_k(scores, classes, k=1):
    precisions = []

    for i, score_row in enumerate(scores):
        sample_class = classes[i]
        top_k_indices = np.argsort(score_row)[-k:][::-1]  # Get top k indices
        correct_retrievals = sum(classes[idx] == sample_class for idx in top_k_indices)
        precision = correct_retrievals / k
        precisions.append(precision)

    return np.nanmean(precisions)

