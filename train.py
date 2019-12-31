import torch
import torch.nn as nn
from torch.autograd import Variable
from module import subsequent_mask


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return (loss * norm).item()


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def greedy_decode(tree_transformer_model, batch, max_len, start_pos):
    memory = tree_transformer_model.encode(batch.code,
                                           batch.re_par_ids,
                                           batch.re_bro_ids,
                                           batch.par_matrix,
                                           batch.bro_matrix)
    ys = torch.ones(1, 1).fill_(start_pos).type_as(batch.code.data)
    for i in range(max_len - 1):
        #  memory, code_mask, comment, comment_mask
        out = tree_transformer_model.decode(memory, batch.code_mask,
                                            Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(batch.code.data)))
        prob = tree_transformer_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(batch.code.data).fill_(next_word)], dim=1)
    return ys


class Batch:
    def __init__(self,
                 code,
                 par_matrix,
                 bro_matrix,
                 re_par_ids,
                 re_bro_ids,
                 comments=None,
                 pad=0):
        # 加载入gpu
        if torch.cuda.is_available():
            code = code.cuda()
            par_matrix = par_matrix.cuda()
            bro_matrix = bro_matrix.cuda()
            re_par_ids = re_par_ids.cuda()
            re_bro_ids = re_bro_ids.cuda()
            if comments is not None:
                comments = comments.cuda()

        self.code = code
        # code_mask用于解码时用
        self.code_mask = (code != pad).unsqueeze(-2)
        self.par_matrix = par_matrix
        self.bro_matrix = bro_matrix
        self.re_par_ids = re_par_ids
        self.re_bro_ids = re_bro_ids
        if comments is not None:
            self.comments = comments[:, :-1]
            self.predicts = comments[:, 1:]
            self.comment_mask = self.make_std_mask(self.comments, pad)
            # 训练时的有效预测个数
            self.ntokens = (self.predicts != pad).data.sum()

    @staticmethod
    def make_std_mask(comment, pad):
        comment_mask = (comment != pad).unsqueeze(-2)
        tgt_mask = comment_mask & Variable(
            subsequent_mask(comment.size(-1)).type_as(comment_mask.data))
        return tgt_mask
