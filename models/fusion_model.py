"""Multi-modal fusion models: concatenation, gated, and attention-based fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModel(nn.Module):
    """Early concatenation fusion for multi-modal Alzheimer's prediction.

    Concatenates MRI and Audio embeddings, passes through FC layers,
    outputs a single scalar severity score.

    Also supports unimodal mode (single embedding input) for ablation studies.
    """

    def __init__(self, embed_dim=256, hidden_dims=(256, 128), dropout=0.3,
                 num_modalities=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        input_dim = embed_dim * num_modalities

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        # Final projection to severity score
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, *embeddings):
        """Compute severity score from one or more embeddings.

        Args:
            *embeddings: one or more (B, embed_dim) tensors

        Returns:
            severity_score: (B, 1) continuous severity
        """
        x = torch.cat(embeddings, dim=1)
        return self.network(x)


class GatedFusionModel(nn.Module):
    """Gated fusion with learnable modality importance weights.

    Each modality gets a scalar gate that controls its contribution
    to the fused representation before being passed to the classifier.
    """

    def __init__(self, embed_dim=256, hidden_dims=(256, 128), dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim

        # Gating networks: one per modality
        self.gate_mri = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.gate_audio = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Shared classifier on gated concatenation
        input_dim = embed_dim * 2
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, mri_emb, audio_emb):
        """Compute severity score with gated fusion.

        Args:
            mri_emb: (B, embed_dim) MRI embedding
            audio_emb: (B, embed_dim) Audio embedding

        Returns:
            severity_score: (B, 1) continuous severity
        """
        g_mri = self.gate_mri(mri_emb)      # (B, 1)
        g_audio = self.gate_audio(audio_emb)  # (B, 1)

        fused = torch.cat([g_mri * mri_emb, g_audio * audio_emb], dim=1)
        return self.classifier(fused)


class AttentionFusionModel(nn.Module):
    """Cross-modal attention fusion.

    Uses a simple cross-attention mechanism where each modality
    attends to the other to produce an enriched representation.
    """

    def __init__(self, embed_dim=256, num_heads=4, hidden_dims=(256, 128),
                 dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim

        # Cross-attention: MRI queries attend to Audio keys/values and vice versa
        self.cross_attn_mri = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_audio = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm_mri = nn.LayerNorm(embed_dim)
        self.norm_audio = nn.LayerNorm(embed_dim)

        # Classifier on enriched concatenation
        input_dim = embed_dim * 2
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, mri_emb, audio_emb):
        """Compute severity score with cross-modal attention fusion.

        Args:
            mri_emb: (B, embed_dim) MRI embedding
            audio_emb: (B, embed_dim) Audio embedding

        Returns:
            severity_score: (B, 1) continuous severity
        """
        # Reshape to (B, 1, embed_dim) for attention (single-token sequences)
        mri_seq = mri_emb.unsqueeze(1)
        audio_seq = audio_emb.unsqueeze(1)

        # Cross-attention: MRI attends to Audio
        mri_enriched, _ = self.cross_attn_mri(mri_seq, audio_seq, audio_seq)
        mri_enriched = self.norm_mri(mri_seq + mri_enriched).squeeze(1)

        # Cross-attention: Audio attends to MRI
        audio_enriched, _ = self.cross_attn_audio(audio_seq, mri_seq, mri_seq)
        audio_enriched = self.norm_audio(audio_seq + audio_enriched).squeeze(1)

        fused = torch.cat([mri_enriched, audio_enriched], dim=1)
        return self.classifier(fused)
