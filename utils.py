import os
import math
import torch
import config
from torch import einsum
from einops import rearrange
from torch.autograd.function import Function
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def calculate_metrics(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    y_true_bin = torch.nn.functional.one_hot(
        torch.tensor(y_true), num_classes=10).numpy()
    roc_auc = roc_auc_score(y_true_bin, y_pred_proba,
                            average='weighted', multi_class='ovr')

    return accuracy, balanced_accuracy, f1, roc_auc


def save_metrics(filename, epoch, phase, loss, accuracy, balanced_accuracy, f1, roc_auc):
    os.makedirs('logs', exist_ok=True)
    with open(f'logs/{filename}', 'a') as f:
        f.write(f"Epoch: {epoch}, Phase: {phase}\n")
        f.write(f"  Loss: {loss:.4f}\n")
        f.write(f"  Accuracy: {accuracy:.4f}\n")
        f.write(f"  Balanced Accuracy: {balanced_accuracy:.4f}\n")
        f.write(f"  F1 Score: {f1:.4f}\n")
        f.write(f"  ROC AUC: {roc_auc:.4f}\n\n")


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class FlashAttentionFunction(Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
        """ Algorithm 1 in the v2 paper """

        device = q.device
        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((*q.shape[:-1], 1), device=device)
        all_row_maxes = torch.full(
            (*q.shape[:-1], 1), max_neg_value, device=device)

        scale = (q.shape[-1] ** -0.5)

        num_row_tiles = math.ceil(q.shape[-2] / q_bucket_size)
        num_col_tiles = math.ceil(k.shape[-2] / k_bucket_size)

        if exists(mask) and mask.ndim == 2:
            mask = rearrange(mask, 'b n -> b 1 1 n')

        if not exists(mask):
            col_masks = (None,) * num_col_tiles
            mask = (col_masks,) * num_row_tiles
        else:
            mask = (
                (mask,) * num_row_tiles) if mask.shape[-2] == 1 else mask.split(q_bucket_size, dim=-2)
            mask = tuple(((row_mask,) * num_col_tiles) if row_mask.shape[-1] == 1 else row_mask.split(
                k_bucket_size, dim=-1) for row_mask in mask)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            mask,
            all_row_sums.split(q_bucket_size, dim=-2),
            all_row_maxes.split(q_bucket_size, dim=-2),
        )

        for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
                row_mask
            )

            for k_ind, (kc, vc, col_mask) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum(
                    '... i d, ... j d -> ... i j', qc, kc) * scale

                if exists(col_mask):
                    attn_weights.masked_fill_(~col_mask, max_neg_value)

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones(
                        (qc.shape[-2], kc.shape[-2]), dtype=torch.bool, device=device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                block_row_maxes = attn_weights.amax(dim=-1, keepdims=True)
                new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                exp_weights = torch.exp(attn_weights - new_row_maxes)

                if exists(col_mask):
                    exp_weights.masked_fill_(~col_mask, 0.)

                block_row_sums = exp_weights.sum(
                    dim=-1, keepdims=True).clamp(min=config.EPSILON)

                exp_values = einsum(
                    '... i j, ... j d -> ... i d', exp_weights, vc)

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)

                new_row_sums = exp_row_max_diff * row_sums + block_row_sums

                oc.mul_(exp_row_max_diff).add_(exp_values)

                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)

            oc.div_(row_sums)

        lse = all_row_sums.log() + all_row_maxes

        ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
        ctx.save_for_backward(q, k, v, o, lse)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        """ Algorithm 2 in the v2 paper """

        causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
        q, k, v, o, lse = ctx.saved_tensors

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            do.split(q_bucket_size, dim=-2),
            mask,
            lse.split(q_bucket_size, dim=-2),
            dq.split(q_bucket_size, dim=-2)
        )

        for ind, (qc, oc, doc, row_mask, lsec, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
                dk.split(k_bucket_size, dim=-2),
                dv.split(k_bucket_size, dim=-2),
                row_mask
            )

            for k_ind, (kc, vc, dkc, dvc, col_mask) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum(
                    '... i d, ... j d -> ... i j', qc, kc) * scale

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones(
                        (qc.shape[-2], kc.shape[-2]), dtype=torch.bool, device=device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                p = torch.exp(attn_weights - lsec)

                if exists(col_mask):
                    p.masked_fill_(~col_mask, 0.)

                dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
                dp = einsum('... i d, ... j d -> ... i j', doc, vc)

                D = (doc * oc).sum(dim=-1, keepdims=True)
                ds = p * scale * (dp - D)

                dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
                dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)

        return dq, dk, dv, None, None, None, None