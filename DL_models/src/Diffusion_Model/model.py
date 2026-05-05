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

# -----------------------------
# Waveform Encoder
# -----------------------------
class WaveformEncoder(nn.Module):
    def __init__(self, num_vitals, embed_dim, n_head=2, n_layers=2):
        super().__init__()

        # Conv Encoding
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=3, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=3, padding=2),
            nn.GELU(),
            nn.Conv1d(128, embed_dim, kernel_size=3, stride=2, padding=1),
        )

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

    def forward(self, x): # [B, 1, 3750]
        device = x.device
        
        #Embed Inputs
        x = self.conv(x) # [B, embed_dim, 53]
        x = x.permute(0, 2, 1) # [B, 53, embed_dim]

        # Normalize
        x = self.norm(x) # [B, 53, embed_dim]

        #Add Positional Encoding
        x = self.pos_encoder(x) # [B, 53, embed_dim]

        # Transformer Encoder
        x = self.transformer(x) # [B, 53, embed_dim]

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
    def __init__(self, num_vitals, prediction_length, embed_dim, num_heads, num_layers, waveform_conditioning=False, clinical_conditioning=False, use_pretrained_vital_encoder_weights=False, hr_encoder_pretrained_weights=None, resp_encoder_pretrained_weights=None, spO2_encoder_pretrained_weights=None):
        super().__init__()

        self.num_vitals = num_vitals
        self.prediction_length = prediction_length
        self.waveform_conditioning = waveform_conditioning
        self.clinical_conditioning = clinical_conditioning

        # Vital Encoders
        self.hr_encoder = VitalsEncoder(num_vitals=1, embed_dim=embed_dim)
        self.resp_encoder = VitalsEncoder(num_vitals=1, embed_dim=embed_dim)
        self.spO2_encoder = VitalsEncoder(num_vitals=1, embed_dim=embed_dim)

        if use_pretrained_vital_encoder_weights:
            #HR
            hr_weights = torch.load(hr_encoder_pretrained_weights, map_location="cpu", weights_only=False)
            hr_vital_encoder_weights = {}
            for key in hr_weights.keys():
                if 'vital_encoder' in key:
                    weight_key = key[14:]
                    hr_vital_encoder_weights[weight_key] = hr_weights[key]
                     
            self.hr_encoder.load_state_dict(hr_vital_encoder_weights)

            #RESP
            resp_weights = torch.load(resp_encoder_pretrained_weights, map_location="cpu", weights_only=False)
            resp_vital_encoder_weights = {}
            for key in resp_weights.keys():
                if 'vital_encoder' in key:
                    weight_key = key[14:]
                    resp_vital_encoder_weights[weight_key] = resp_weights[key]
                     
            self.resp_encoder.load_state_dict(resp_vital_encoder_weights)

            #SpO2
            spO2_weights = torch.load(spO2_encoder_pretrained_weights, map_location="cpu", weights_only=False)
            spO2_vital_encoder_weights = {}
            for key in spO2_weights.keys():
                if 'vital_encoder' in key:
                    weight_key = key[14:]
                    spO2_vital_encoder_weights[weight_key] = spO2_weights[key]
                     
            self.spO2_encoder.load_state_dict(spO2_vital_encoder_weights)

        #Vital Fusion
        self.cross_vital_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2  # keep shallow
        )

        # Waveform Encoders
        if self.waveform_conditioning:
            self.ecg_encoder = WaveformEncoder(num_vitals=1, embed_dim=embed_dim)
            self.resp_wave_encoder = WaveformEncoder(num_vitals=1, embed_dim=embed_dim)
            self.pleth_encoder = WaveformEncoder(num_vitals=1, embed_dim=embed_dim)

            #Waveform Fusion
            self.cross_waveform_fusion = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=2  # keep shallow
            )

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

    def forward(self, noisy_future, past_values, past_masks, past_deltas, t, waveform_values=None, clinical_embeddings=None):
        '''
        noisy_future: [B, 10, 3]
        past_values: [B, 60, 3]
        past_masks:  [B, 60, 3]
        past_deltas: [B, 60, 3]
        t:      [B] diffusion timestep   
        waveform_values: [B, 3750, 3]
        clinical_embeddings: [B, N, 256]
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
        hr_tokens = self.hr_encoder(past_values[:, :, 0:1], past_masks[:, :, 0:1], past_deltas[:, :, 0:1]) # [B, 60, embed_dim]
        resp_tokens = self.resp_encoder(past_values[:, :, 1:2], past_masks[:, :, 1:2], past_deltas[:, :, 1:2]) # [B, 60, embed_dim]
        spO2_tokens = self.spO2_encoder(past_values[:, :, 2:3], past_masks[:, :, 2:3], past_deltas[:, :, 2:3])# [B, 60, embed_dim]
        vital_tokens = torch.cat([hr_tokens, resp_tokens, spO2_tokens], dim=1)# [B, 180, embed_dim]
        vital_tokens = self.cross_vital_fusion(vital_tokens) # [B, 180, embed_dim]
        
        if self.waveform_conditioning:
            ecg_tokens = self.ecg_encoder(waveform_values[:, :, 0:1].permute(0, 2, 1)) # [B, 53, embed_dim]
            resp_wave_tokens = self.resp_wave_encoder(waveform_values[:, :, 1:2].permute(0, 2, 1)) # [B, 53, embed_dim]
            pleth_tokens = self.pleth_encoder(waveform_values[:, :, 2:3].permute(0, 2, 1))# [B, 53, embed_dim]
            waveform_tokens = torch.cat([ecg_tokens, resp_wave_tokens, pleth_tokens], dim=1)# [B, 159, embed_dim]
            waveform_tokens = self.cross_waveform_fusion(waveform_tokens) # [B, 159, embed_dim]

            conditioning_tokens = torch.cat([vital_tokens, waveform_tokens], dim=1) # [B, 339, embed_dim]
        elif self.clinical_conditioning:
            conditioning_tokens = torch.cat([vital_tokens, clinical_embeddings], dim=1) # [B, 181, embed_dim]
        else:
            conditioning_tokens = vital_tokens

        #Diffusion Blocks
        for block in self.diff_layers:
            future_tokens = block(future_tokens, conditioning_tokens) # [B, 10, embed_dim]

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


