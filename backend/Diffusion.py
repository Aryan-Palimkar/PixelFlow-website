import torch.nn as nn
import torch.nn.functional as F
import torch

class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        assert in_channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Conv2d(in_channels, in_channels, 1)
        self.to_k = nn.Conv2d(in_channels, in_channels, 1)
        self.to_v = nn.Conv2d(in_channels, in_channels, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q = self.to_q(x).view(B, self.num_heads, self.head_dim, N).permute(0,1,3,2)
        k = self.to_k(x).view(B, self.num_heads, self.head_dim, N)
        v = self.to_v(x).view(B, self.num_heads, self.head_dim, N).permute(0,1,3,2)
        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0,1,3,2).reshape(B, C, H, W)
        return x + self.to_out(out)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=512, norm_groups=8):
        super().__init__()
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(norm_groups, out_ch)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(norm_groups, out_ch)
        self.act2 = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = x + self.time_proj(t_emb)[:, :, None, None]
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x + h

class UNet128(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, time_emb_dim=512):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(128, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.init = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.down1 = nn.Conv2d(64, 128, 4, 2, 1); self.rb1 = ResBlock(128, 128, time_emb_dim)
        self.down2 = nn.Conv2d(128, 256, 4, 2, 1); self.rb2 = ResBlock(256, 256, time_emb_dim)
        self.down3 = nn.Conv2d(256, 512, 4, 2, 1); self.rb3 = ResBlock(512, 512, time_emb_dim)
        self.down4 = nn.Conv2d(512, 1024, 4, 2, 1); self.rb4 = ResBlock(1024, 1024, time_emb_dim)

        self.mid1 = ResBlock(1024, 1024, time_emb_dim)
        self.mid_attn = SelfAttention(1024, num_heads=8)
        self.mid2 = ResBlock(1024, 1024, time_emb_dim)

        self.up4 = nn.ConvTranspose2d(1024, 512, 4, 2, 1); self.urb4 = ResBlock(1024, 512, time_emb_dim)
        self.attn4 = SelfAttention(512, num_heads=8)
        self.up3 = nn.ConvTranspose2d(512, 256, 4, 2, 1); self.urb3 = ResBlock(512, 256, time_emb_dim)
        self.attn3 = SelfAttention(256, num_heads=8)
        self.up2 = nn.ConvTranspose2d(256, 128, 4, 2, 1); self.urb2 = ResBlock(256, 128, time_emb_dim)
        self.attn2 = SelfAttention(128, num_heads=4)
        self.up1 = nn.ConvTranspose2d(128, 64, 4, 2, 1); self.urb1 = ResBlock(128, 64, time_emb_dim)
        self.attn1 = SelfAttention(64, num_heads=4)

        self.final = nn.Conv2d(64, out_ch, 3, padding=1)

    def sinusoidal_embedding(self, t, dim=128):
        half = dim // 2
        freq = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half, device=t.device) / (half - 1))
        emb = t[:, None] * freq[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(self, x, t):
        t = t.to(x.device).float()
        temb = self.time_embed(self.sinusoidal_embedding(t))
        
        x0 = self.init(x)
        # down
        d1 = self.down1(x0); d1 = self.rb1(d1, temb)
        d2 = self.down2(d1); d2 = self.rb2(d2, temb)
        d3 = self.down3(d2); d3 = self.rb3(d3, temb)
        d4 = self.down4(d3); d4 = self.rb4(d4, temb)
        # bottleneck
        m = self.mid1(d4, temb)
        m = self.mid_attn(m)
        m = self.mid2(m, temb)
        # up
        u4 = self.up4(m); u4 = torch.cat([u4, d3], dim=1); u4 = self.urb4(u4, temb); u4 = self.attn4(u4)
        u3 = self.up3(u4); u3 = torch.cat([u3, d2], dim=1); u3 = self.urb3(u3, temb) 
        u2 = self.up2(u3); u2 = torch.cat([u2, d1], dim=1); u2 = self.urb2(u2, temb) 
        u1 = self.up1(u2); u1 = torch.cat([u1, x0], dim=1); u1 = self.urb1(u1, temb)
        return self.final(u1)

class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def cosine_beta_schedule(timesteps, s=0.008, device='cpu'):
    steps = torch.arange(timesteps, dtype=torch.float32, device=device) / timesteps
    f_t = torch.cos((steps + s) / (1 + s) * torch.pi / 2) ** 2
    betas = torch.clamp(1 - f_t / f_t.roll(1), min=0.0001, max=0.9999)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return betas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

betas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = cosine_beta_schedule(1000, device='cpu')

def sample(model, n_samples=4, device='cpu', img_size=64, channels=3, sample_steps=50, eta=0.0):
    model.eval()
    with torch.no_grad():
        T = betas.shape[0]
        step_indices = torch.linspace(T-1, 0, steps=sample_steps+1, dtype=torch.long, device=device).round().long()
        timesteps = step_indices[:-1]
        next_timesteps = step_indices[1:]

        x = torch.randn(n_samples, channels, img_size, img_size, device=device)

        for i, (t, t_next) in enumerate(zip(timesteps, next_timesteps)):
            t_tensor = torch.full((n_samples,), t, dtype=torch.long, device=device)
            
            predicted_noise = model(x, t_tensor)

            alpha_bar_t = alphas_cumprod[t]
            alpha_bar_t_next = alphas_cumprod[t_next]
            sqrt_alpha_bar_t = sqrt_alphas_cumprod[t]
            sqrt_alpha_bar_t_next = sqrt_alphas_cumprod[t_next]
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_cumprod[t]

            x_0_pred = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
            x_0_pred = torch.clamp(x_0_pred, -1., 1.)

            sigma_t = eta * torch.sqrt((1 - alpha_bar_t_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_next))
            sigma_t = torch.clamp(sigma_t, 0., 1.)
            noise = torch.randn_like(x) * sigma_t if t_next > 0 else torch.zeros_like(x)

            x = sqrt_alpha_bar_t_next * x_0_pred + torch.sqrt(1 - alpha_bar_t_next - sigma_t**2) * predicted_noise + noise
            x = torch.nan_to_num(x, nan=0.0)

        x = torch.clamp(x, -1., 1.)
        print(f"Sampled images shape: {x.shape}, min: {x.min().item()}, max: {x.max().item()}")
        return x

model = model = UNet128()
ema = EMA(model)