import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self, trg_vocab_size, embed_size, num_layers,
        heads, forward_expansion, dropout, device, max_length
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


class TransformerDecoder(nn.Module):
    def __init__(
            self, vocab_size, pad_idx, embed_size=512,
            num_layers=6, forward_expansion=4, heads=8, dropout=0,
            device=torch.device("cuda"), max_length=100
    ):
        super().__init__()
        self.decoder = Decoder(
            vocab_size, embed_size, num_layers, heads,
            forward_expansion, dropout, device, max_length,
        ).to(device)

        self.pad_idx = pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_cap_mask(self, cap):
        N, cap_len = cap.shape
        cap_mask = torch.tril(torch.ones((cap_len, cap_len))).expand(N, 1, cap_len, cap_len)
        return cap_mask.to(self.device)

    def forward(self, features, captions):
        features = features.reshape(features.shape[0], features.shape[1], -1).transpose(1, 2)
        captions = torch.transpose(captions, 0, 1)
        captions_mask = self.make_cap_mask(captions)

        output = self.decoder(captions, features, None, captions_mask)
        return output.transpose(0, 1)


if __name__ == "__main__":
    from TransformerEncoder import TransformerEncoder

    device = torch.device("cuda")

    images = torch.randn((2, 3, 256, 256)).to(device)
    captions = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]).to(device)
    captions = torch.transpose(captions, 0, 1)

    encoder = TransformerEncoder(patch_size=16, dim=512).to(device)
    decoder = TransformerDecoder(vocab_size=100, pad_idx=0, embed_size=512, num_layers=1).to(device)

    features = encoder(images)
    print("captions main:", captions.shape)
    output = decoder(features, captions[:-1])
    print("main output:", output.shape)

    # should be: [21, 32, 2994], [21, 32] ----- [32] is batch size
    print("output & captions:", output.shape, captions[1:].shape)

    # should be: [768, 2994], [768]
    print("loss:", output.reshape(-1, output.shape[2]).shape, captions[1:].reshape(-1).shape)

