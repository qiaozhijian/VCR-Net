import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None, is_src=False, overlap2=0.75):
    """
        single attention calculation

        Args:
            query: vector Q
            key: vector K
            value: vector V
            mask: shows which grid would have a score of -1e9, matrix
            dropout: not used
            is_src: is source point net
            overlap2: ratio of the point net after removing the non-relevant points. Only considered when is_src=True

        Returns:
            self-attention result, softmax result
        """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)  # weight obtained from softmax
    if is_src:
        batch_size, n_head, num_points_key, n_dims = key.size()
        _, _, num_points_query, _ = query.size()
        idx_base = torch.arange(0, batch_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) \
                       .view(-1, 1, 1, 1) * num_points_key
        scoresColSum = torch.sum(p_attn, dim=[1, 2], keepdim=True)
        tgtK = int(num_points_key * overlap2)
        idxColSum = scoresColSum.topk(k=tgtK, dim=-1)[1]
        idxColSum = idxColSum + idx_base
        idxColSum = idxColSum.view(-1)
        if torch.cuda.is_available():
            mask2 = torch.full((batch_size, num_points_key, num_points_query), fill_value=0, dtype=torch.long).cuda()
        else:
            mask2 = torch.full((batch_size, num_points_key, num_points_query), fill_value=0, dtype=torch.long).cpu()
        mask2.view(batch_size * num_points_key, num_points_query)[idxColSum, :] = 1
        mask2 = mask2.transpose(-2, -1).contiguous().view(batch_size, 1, num_points_query, num_points_key).repeat(
            (1, n_head, 1, 1))
        scores = scores.masked_fill(mask2 == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)

    return torch.matmul(p_attn, value), p_attn


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  # nn.Sequential()
        self.tgt_embed = tgt_embed  # nn.Sequential()
        self.generator = generator  # nn.Sequential()

    def forward(self, src, tgt, src_mask, tgt_mask, src_tgt=True):
        "Take in and process masked src and target sequences."
        re = self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

        return re

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class Generator(nn.Module):
    def __init__(self, emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(emb_dims, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, x):
        x = self.nn(x.max(dim=1)[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size  # 512

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, is_src=False, overlap2=0.75, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # 512//4
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None
        self.is_src = is_src
        self.overlap2 = overlap2

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # query=key=value=[B,1024,512]
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]
        # [B,4,1024,128]=q=k=v
        # 2) Apply attention on all the projected vectors in batch.

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout, is_src=self.is_src, overlap2=self.overlap2)

        self.attn = torch.sum(self.attn, dim=1)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.emb_dims = args.emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.ff_dims = args.ff_dims
        self.n_heads = args.n_heads
        self.overlap2 = args.overlap2  # the amount of points that overlap between two PCs
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims, is_src=False)
        if args.partial:
            src_attn = MultiHeadedAttention(self.n_heads, self.emb_dims, is_src=True, overlap2=self.overlap2)
        else:
            src_attn = MultiHeadedAttention(self.n_heads, self.emb_dims, is_src=False)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(src_attn), c(ff), self.dropout),
                                            self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None, src_tgt=False).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None, src_tgt=True).transpose(2, 1).contiguous()

        return src_embedding, tgt_embedding
