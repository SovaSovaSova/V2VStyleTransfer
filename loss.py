import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use mean/std calc


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class CCPLLoss(nn.Module):
    """
    Contrastive Coherence Preserving Loss (ECCV 2022)
    """

    def __init__(self, temperature=0.07):
        super(CCPLLoss, self).__init__()
        self.temperature = temperature

    def forward(self, current_feat, warped_prev_feat, mask=None):
        """
        feat: [B, C, H, W]
        """
        B, C, H, W = current_feat.shape

        # Normalize features for cosine similarity
        current_feat = F.normalize(current_feat, dim=1)
        warped_prev_feat = F.normalize(warped_prev_feat, dim=1)

        # Flatten spatial dimensions -> [B, C, N]
        curr_flat = current_feat.view(B, C, -1)
        warped_flat = warped_prev_feat.view(B, C, -1)

        # 1. Positive Logits: Cosine similarity between corresponding points
        pos_logits = torch.sum(curr_flat * warped_flat, dim=1)  # [B, N]
        pos_logits = pos_logits / self.temperature

        # 2. Negative Logits: Similarity to other pixels
        # Simplified: Randomly shuffle warped features as negatives
        # This is a lightweight approximation of the full CCPL negative sampling
        neg_idx = torch.randperm(curr_flat.shape[2])
        shuffled_warped = warped_flat[:, :, neg_idx]
        neg_logits = torch.sum(
            curr_flat * shuffled_warped, dim=1) / self.temperature

        # InfoNCE Loss
        loss = torch.log(1 + torch.exp(neg_logits - pos_logits))

        if mask is not None:
            # Handle mask shape [B, C, H, W] -> [B, H*W]
            if mask.dim() == 4:
                mask = mask[:, 0, :, :]  # Take first channel
            mask_flat = mask.reshape(B, -1)

            # Safe masking
            loss = loss * mask_flat
            return loss.sum() / (mask_flat.sum() + 1e-8)
        else:
            return loss.mean()
