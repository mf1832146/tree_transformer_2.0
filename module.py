import math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class EncoderDecoder(nn.Module):
    """
        A standard Encoder-Decoder architecture. Base for this and many
        other models.
    """
    def __init__(self, encoder, decoder, code_embed, nl_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.code_embed = code_embed
        self.nl_embed = nl_embed
        self.generator = generator

    def forward(self, code, relative_par_ids, relative_bro_ids, nl, par_mask, bro_mask, code_mask, nl_mask):
        return self.decode(self.encode(code, relative_par_ids, relative_bro_ids, par_mask, bro_mask),
                           code_mask, nl, nl_mask)

    def encode(self, code, relative_par_ids, relative_bro_ids, par_mask, bro_mask):
        return self.encoder(self.code_embed(code), relative_bro_ids, relative_par_ids, par_mask, bro_mask)

    def decode(self, memory, code_mask, nl, nl_mask):
        return self.decoder(self.nl_embed(nl), memory, code_mask, nl_mask)


class RelativePositionEmbedding(nn.Module):
    def __init__(self, d_model, k, num_heads, dropout=0.0):
        """
        生成相对位置信息编码
        :param d_model: 词向量维度
        :param k: 相对位置窗口大小
        :param dropout:
        """
        super(RelativePositionEmbedding, self).__init__()

        self.d_model = d_model
        self.k = k

        self.parent_emb = nn.Embedding(2*k+2, d_model * 2, padding_idx=1)
        self.brother_emb = nn.Embedding(2*k+2, d_model * 2, padding_idx=1)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, relation_type):
        """
        :param inputs: 相对位置矩阵, 即traverse中的relative_parent_ids or
        relative_brother_ids shape [batch_size, max_size, max_size]
        :param relation_type: 'parent' means find relation between parent and child, 'brother' means find relation between brothers
        :return:
        """
        batch_size, max_size = inputs.size(0), inputs.size(1)
        inputs = inputs.unsqueeze(3)
        if relation_type == 'parent':
            position_emb = self.parent_emb(inputs)
        else:
            position_emb = self.brother_emb(inputs)
        position_emb = self.dropout(position_emb)
        position_emb = position_emb.view(batch_size, max_size, max_size, 2, self.d_model)
        k_emb, v_emb = [x.squeeze(3) for x in position_emb.split(1, dim=3)]

        k_emb = k_emb.repeat(1, 1, 1, self.num_heads)
        v_emb = v_emb.repeat(1, 1, 1, self.num_heads)

        k_emb = k_emb.view(-1, max_size, max_size, self.d_model) * math.sqrt(self.d_model)
        v_emb = v_emb.view(-1, max_size, max_size, self.d_model) * math.sqrt(self.d_model)

        return k_emb, v_emb


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N, relative_pos_emb):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        self.relative_pos_emb = relative_pos_emb

    def forward(self, code, relative_par_ids, relative_bro_ids, par_mask, bro_mask):
        par_k_emb, par_v_emb = self.relative_pos_emb(relative_par_ids, 'parent')
        bro_k_emb, bro_v_emb = self.relative_pos_emb(relative_bro_ids, 'brother')

        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                code = layer(code, par_k_emb, par_v_emb, par_mask)
            else:
                code = layer(code, bro_k_emb, bro_v_emb, bro_mask)
        return self.norm(code)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, code, rel_q_emb, rel_v_emb, mask):
        code = self.sublayer[0](code, lambda x: self.self_attn(x, x, x, mask, rel_q_emb, rel_v_emb))
        return self.sublayer[1](code, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


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


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def relative_mul(q, relative):
    """relative position dot product"""
    node_len, dim_per_head = relative.size(2), relative.size(3)
    relative_k = relative.transpose(2, 3).view(-1, dim_per_head, node_len)
    q = q.view(-1, dim_per_head).unsqueeze(1)
    return torch.bmm(q, relative_k).squeeze(1).view(-1, node_len, node_len)


def relative_attn(query, key, value, mask=None, relative_q=None, relative_v=None, dropout=None):
    batch_size, num_heads, length, per_head = query.size()
    query = query.contiguous().view(batch_size * num_heads, length, per_head)
    key = key.contiguous().view(batch_size * num_heads, length, per_head)
    value = value.contiguous().view(batch_size * num_heads, length, per_head)
    mask = mask.repeat(1, num_heads, 1, 1)
    mask = mask.contiguous().view(batch_size * num_heads, length, -1)

    scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(per_head)
    if relative_q is not None:
        scores += relative_mul(query, relative_q)
    if mask is not None:
        # 给需要mask的地方设置一个负无穷（因为接下来要输入到softmax层，如果是0还是会有影响）
        scores = scores.masked_fill(mask == 0, -1e9)
    # 计算softmax
    p_attn = F.softmax(scores, dim=-1)
    # 添加dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 和V做点积
    context = torch.bmm(p_attn, value)
    if relative_v is not None:
        node_len, dim_per_head = relative_v.size(2), relative_v.size(3)
        att_v = p_attn.view(-1, node_len).unsqueeze(1)
        relative_v = relative_v.view(-1, node_len, dim_per_head)
        context_v = torch.bmm(att_v, relative_v).squeeze(1)
        context_v = context_v.view(-1, node_len, dim_per_head)
        context += context_v
    context = context.contiguous().view(batch_size, num_heads, length, per_head)
    return context, p_attn


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, relative_q=None, relative_v=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # x, self.attn = attention(query, key, value, mask=mask,
        #                          dropout=self.dropout)
        if relative_q is not None:
            x, self.attn = relative_attn(query, key, value, mask=mask,
                                         relative_q=relative_q, relative_v=relative_v,
                                         dropout=self.dropout)
        else:
            x, self.attn = attention(query, key, value, mask=mask,
                                     dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(gelu(self.w_1(x))))


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def make_model(code_vocab, nl_vocab, N=6,
               d_model=512, d_ff=2048, k=2, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), 2 * N,
                RelativePositionEmbedding(d_model // h, k, h, dropout)),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        Embeddings(d_model, code_vocab),
        nn.Sequential(Embeddings(d_model, nl_vocab), c(position)),
        Generator(d_model, nl_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model