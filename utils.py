import numpy as np
import torch
import pickle


def load_txt(file_path, skip=0):
    data = np.loadtxt(file_path, dtype=np.long, skiprows=skip)
    data = torch.from_numpy(data)
    return data


def log(msg, file_path):
    with open(file_path, 'a+') as f:
        f.write(msg + '\n')


def load_dict(file_path):
    return pickle.load(file_path)


def load(save_path, max_size, skip=0):
    code = load_txt(save_path + 'code.txt', skip)
    print('loading code finished')
    parent_matrix = load_txt(save_path + 'parent_matrix.txt', skip).view(-1, max_size, max_size)
    print('loading parent_matrix finished')
    brother_matrix = load_txt(save_path + 'brother_matrix.txt', skip).view(-1, max_size, max_size)
    print('loading brother_matrix finished')
    rel_par_ids = load_txt(save_path + 'relative_parents.txt', skip).view(-1, max_size, max_size)
    print('loading rel_par_ids finished')
    rel_bro_ids = load_txt(save_path + 'relative_brothers.txt', skip).view(-1, max_size, max_size)
    print('loading rel_bro_ids finished')
    comments = load_txt(save_path + 'comments.txt', skip)
    print('loading comments finished')

    return code, parent_matrix, brother_matrix, rel_par_ids, rel_bro_ids, comments