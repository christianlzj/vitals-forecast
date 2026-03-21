import torch.nn as nn
import torch
import math

# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FuturePositionalEncoder(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# -----------------------------
# Vitals Encoder
# -----------------------------
class VitalsEncoder(nn.Module):
    def __init__(self, num_vitals, embed_dim, n_head=4, n_layers=4):
        super().__init__()

        # Embedding Projections
        self.value_proj = nn.Linear(num_vitals, embed_dim)
        self.mask_proj = nn.Linear(num_vitals, embed_dim)
        self.delta_proj = nn.Linear(num_vitals, embed_dim)

        #Normalize
        self.norm = nn.LayerNorm(embed_dim)

        #Position Encodings
        self.pos_encoder = PositionalEncoding(embed_dim)

        #Transformer Encoding
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=n_head, 
                dim_feedforward=embed_dim*4,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            ),
            num_layers=n_layers
        )

    def forward(self, values, masks=None, deltas=None):
        device = values.device

        #handle when no masks or deltas given
        if masks is None:
            masks = torch.zeros_like(values).to(device)
            deltas = torch.zeros_like(values).to(device)

        #Embed Inputs
        v = self.value_proj(values) # [B, 60, embed_dim]
        m = self.mask_proj(masks) # [B, 60, embed_dim] 
        d = self.delta_proj(deltas) # [B, 60, embed_dim]

        #Combine and Normalize
        x = v + m + d
        x = self.norm(x) # [B, 60, embed_dim]

        #Add Positional Encoding
        x = self.pos_encoder(x) # [B, 60, embed_dim]

        #Temporal Transformer
        x = self.transformer(x) # [B, 60, embed_dim]

        return x


# -------------------------------------------------
#   Diffusion Timestep Embedding (diffusion step)
# -------------------------------------------------
class DiffusionTimestepEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, t):
        """
        t: (B,)
        """
        half_dim = self.embed_dim // 2

        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, device=t.device) / half_dim
        )

        args = t[:, None] * freqs[None]  # (B, half_dim)

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, embed_dim)

        return self.mlp(emb)



class DiffusionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.norm_3 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, future_tokens, context_tokens):
        #Cross Attention: Query Past with Future
        cross_attn_out, _ = self.cross_attn(
            query=future_tokens,
            key=context_tokens,
            value=context_tokens
        )
        future_tokens = self.norm_1(future_tokens + cross_attn_out) #Add Skip and Normalize

        #Self Attention
        self_attn_out, _ = self.self_attn(
            query=future_tokens,
            key=future_tokens,
            value=future_tokens
        )
        future_tokens = self.norm_2(future_tokens + self_attn_out) #Add Skip and Normalize

        #Feed Forward
        ff_out = self.ff(future_tokens)
        future_tokens = self.norm_3(future_tokens + ff_out) #Add Skip and Normalize

        return future_tokens


class DiffusionForecaster(nn.Module):
    def __init__(self, num_vitals, prediction_length, embed_dim, num_heads, num_layers):
        super().__init__()

        self.num_vitals = num_vitals
        self.prediction_length = prediction_length

        # Vitals Encoder
        self.vitals_encoder = VitalsEncoder(num_vitals, embed_dim)

        # Future projection
        self.future_proj = nn.Linear(num_vitals, embed_dim)

        #Future Position Encodings
        self.future_pos_encoder = FuturePositionalEncoder(embed_dim)

        #Diffusion Timestep Embedding
        self.diff_ts_encoder = DiffusionTimestepEmbedding(embed_dim)

        #Diffusion Layers
        self.diff_layers = nn.Sequential(
            *[
                DiffusionBlock(embed_dim, num_heads) for _ in range(num_layers)
            ]
        )

        #Prediction Head
        self.pred_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, num_vitals) #predict noise
        )

    def forward(self, noisy_future, past_values, past_masks, past_deltas, t):
        '''
        noisy_future: [B, 10, 3]
        past_values: [B, 60, 3]
        past_masks:  [B, 60, 3]
        past_deltas: [B, 60, 3]
        t:      [B] diffusion timestep   
        '''

        B = past_values.shape[0]

        #Encode Noisy Future Values
        future_tokens = self.future_proj(noisy_future) # [B, 10, embed_dim]

        #Add Positional Encodings
        future_tokens = self.future_pos_encoder(future_tokens) # [B, 10, embed_dim]

        #Add Diffusion Timestep Embedding
        diff_ts_embed = self.diff_ts_encoder(t) # [B, embed_dim]
        future_tokens = future_tokens + diff_ts_embed.unsqueeze(1) # [B, 10, embed_dim]

        #Encode Vitals
        vital_tokens = self.vitals_encoder(past_values, past_masks, past_deltas) # [B, 60, embed_dim]


        #Diffusion Blocks
        for block in self.diff_layers:
            future_tokens = block(future_tokens, vital_tokens) # [B, 10, embed_dim]

        #Prediction Head
        x = self.pred_head(future_tokens) # [B, 10, 3]

        return x


# attn = nn.MultiheadAttention(
#     embed_dim=128,
#     num_heads=8,
#     batch_first=True
# )

# output,_ = attn(
#     query=future_tokens,
#     key=context_tokens,
#     value=context_tokens,
#     key_padding_mask=context_mask
# )


