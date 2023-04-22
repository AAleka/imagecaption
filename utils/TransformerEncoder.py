import torch
from torch import nn
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.fn(x, **kwargs) + x
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class LSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_ratio, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, LSA(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_ratio * dim, dropout=dropout)))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, patch_size, img_channels=3, dim=256, depth=1, heads=4, dim_head=64, mlp_ratio=4, drop=0.):
        super(TransformerEncoder, self).__init__()
        self.dim = dim
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(img_channels, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.TransformerEncoder = TransformerBlock(dim, depth, heads, dim_head, mlp_ratio, drop)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)

        self.output = nn.Sequential(
            # nn.Conv2d(dim, dim, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
        )

    def forward(self, x):
        if x.shape[2] % self.patch_size != 0 or x.shape[3] % self.patch_size != 0:
            raise ValueError('Image size must be divisible by patch size.')

        N = x.shape[0]
        dim1 = x.shape[2] // self.patch_size
        dim2 = x.shape[3] // self.patch_size

        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = self.TransformerEncoder(x).transpose(1, 2).view(N, self.dim, dim1, dim2)
        x = self.output(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = torch.randn((32, 3, 256, 256)).to(device)
    encoder = TransformerEncoder(patch_size=16, dim=512).to(device)

    features = encoder(img)
    print(features.shape)
