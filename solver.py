from pytorch_pretrained_bert import BertAdam

from module import make_model
import time
import os
from torch.utils.data import DataLoader
from dataset import TreeDataSet
from utils import log, load_dict
from train import *


class Solver:
    def __init__(self, args):
        self.args = args

        self.model_dir = self.make_save_dir(args.model_dir)

        self.model = make_model(code_vocab=self.args.code_vocab_size,
                                nl_vocab=self.args.comment_vocab_size,
                                N=self.args.num_layers,
                                d_model=self.args.model_dim,
                                d_ff=self.args.ffn_dim,
                                k=self.args.k,
                                h=self.args.num_heads,
                                dropout=self.args.dropout)

        if torch.cuda.is_available:
            self.model = self.model.cuda()

    def train(self):
        if self.args.load:
            path = os.path.join(self.model_dir, 'model.pth')
            self.model.load_state_dict(torch.load(path)['state_dict'])

        tt = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                ttt = 1
                for s in param.data.size():
                    ttt *= s
                tt += ttt
        print('total param num:', tt)

        print('Loading training data...')

        train_data_set = TreeDataSet(self.args.train_data_set, self.args.code_max_len, skip=self.args.skip_num)
        test_data_set = TreeDataSet(self.args.test_data_set, self.args.code_max_len, skip=7860)

        train_loader = DataLoader(dataset=train_data_set, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_data_set, batch_size=1, shuffle=False)

        print('load training data finished')

        #model_opt = NoamOpt(self.model.code_embed.d_model, 1, 2000,
        #                    torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        model_opt = NoamOpt(self.model.code_embed.d_model, 1, 2000,
                            BertAdam(self.model.parameters(), lr=1e-4))

        criterion = LabelSmoothing(size=self.args.comment_vocab_size,
                                   padding_idx=0, smoothing=0.1)
        criterion = criterion.cuda()
        loss_compute = SimpleLossCompute(self.model.generator, criterion, model_opt)

        total_loss = []

        for step in range(self.args.num_step):
            self.model.train()

            start = time.time()
            step_loss = run_epoch(step, train_loader, self.model, loss_compute)
            elapsed = time.time() - start
            print('----------epoch: %d end, total loss= %f , train_time= %f Sec -------------' % (step, step_loss, elapsed))
            total_loss.append(step_loss)
            print('saving!!!!')

            model_name = 'model.pth'
            state = {'epoch': step, 'state_dict': self.model.state_dict()}
            torch.save(state, os.path.join(self.model_dir, model_name))
            # test
            self.model.eval()
            self.test(test_loader)

        print('training process end, total_loss is =', total_loss)

    def test(self, data_set_loader=None):
        if self.args.load:
            path = os.path.join(self.model_dir, 'model.pth')
            self.model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['state_dict'])

        if data_set_loader is None:
            data_set = TreeDataSet(self.args.test_data_set, self.args.code_max_len, skip=7860)
            data_set_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False)

        nl_i2w = load_dict(open('./data/nl_i2w.pkl', 'rb'))
        nl_w2i = load_dict(open('./data/nl_w2i.pkl', 'rb'))

        self.model.eval()
        log('_____贪心验证——end_______', './train_model/test.txt')
        for i, data_batch in enumerate(data_set_loader):
            code, par_matrix, bro_matrix, rel_par_ids, rel_bro_ids, comments = data_batch
            batch = Batch(code, par_matrix, bro_matrix, rel_par_ids, rel_bro_ids, None)
            log('Comment:' + ' '.join(nl_i2w[c.item()] for c in comments[0]), './train_model/test.txt')
            start_pos = nl_w2i['<s>']
            predicts = greedy_decode(self.model, batch, self.args.comment_max_len, start_pos)
            log('Predict:' + ' '.join(nl_i2w[c.item()] for c in predicts[0]), './train_model/test.txt')
        log('_____贪心验证——end_______', './train_model/test.txt')

    @staticmethod
    def make_save_dir(save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_dir


def run_epoch(epoch, data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, data_batch in enumerate(data_iter):
        code, par_matrix, bro_matrix, rel_par_ids, rel_bro_ids, comments = data_batch
        batch = Batch(code, par_matrix, bro_matrix, rel_par_ids, rel_bro_ids, comments)
        out = model.forward(batch.code,
                            batch.re_par_ids, batch.re_bro_ids,
                            batch.comments,
                            batch.par_matrix, batch.bro_matrix,
                            batch.code_mask, batch.comment_mask)
        loss = loss_compute(out, batch.predicts, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens.item()
        tokens += batch.ntokens.item()
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch %d Step: %d Loss: %f Tokens per Sec: %f" %
                  (epoch, i, loss / batch.ntokens.item(), tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens
