import argparse
from solver import Solver


def parse():
    parser = argparse.ArgumentParser(description='tree transformer')
    parser.add_argument('-model_dir', default='train_model', help='output model weight dir')
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-num_step', type=int, default=250)
    parser.add_argument('-data_dir', default='./data')
    parser.add_argument('-load', action='store_true', help='load pretrained model')
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-train_data_set', default='./data/tree/train/')
    parser.add_argument('-valid_data_set', default='./data/tree/valid/')
    parser.add_argument('-test_data_set', default='./data/tree/test/')
    parser.add_argument('-skip_num', default=0, type=int)
    parser.add_argument('-pre_trained_path', default='./data/nl_vocab.txt')
    parser.add_argument('-use_pre_trained_emb', default=False)

    parser.add_argument('-code_vocab_size', type=int, default=31131, help='code vocab size')
    parser.add_argument('-code_max_len', type=int, default=100, help='max length of code')
    parser.add_argument('-comment_vocab_size', type=int, default=27596, help='comment vocab size')
    parser.add_argument('-comment_max_len', type=int, default=100, help='comment max length')
    parser.add_argument('-relative_pos', type=bool, default=True, help='use relative position')
    parser.add_argument('-k', type=int, default=5, help='relative window size')
    parser.add_argument('-num_layers', type=int, default=3, help='layer num')
    parser.add_argument('-model_dim', type=int, default=100)
    parser.add_argument('-num_heads', type=int, default=4)
    parser.add_argument('-ffn_dim', type=int, default=512)
    parser.add_argument('-dropout', type=float, default=0.2)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)

    if args.train:
        solver.train()
    elif args.test:
        solver.test()
