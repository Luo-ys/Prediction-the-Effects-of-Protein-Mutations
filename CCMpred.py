#from https://github.com/luoyunan/ECNet
import pathlib
import numpy as np
import pandas as pd
import msgpack
import vocab

def all_sequence_pairwise_profile(args):
    e, index_encoded = args
    N, L = index_encoded.shape
    encoding = np.zeros((N, L, L))
    for k in range(N):
        for i in range(L - 1):
            a = index_encoded[k, i]
            for j in range(i + 1, L):
                b = index_encoded[k, j]
                encoding[k, i, j] = e[i, j, a, b]
        encoding[k] += encoding[k].T
    return encoding


def all_sequence_singleton_profile(args):
    e, index_encoded = args
    N, L = index_encoded.shape
    encoding = np.zeros((N, L, 1))
    for k in range(N):
        for i in range(L):
            a = index_encoded[k, i]
            encoding[k, i] = e[i, a]
    return encoding


class CCMPredEncoder(object):
    def __init__(self, brawfile, seq_len=None):
        self.seq_len = seq_len
        brawfile = pathlib.Path(brawfile)
        self.vocab_index = vocab.CCMPRED_AMINO_ACID_INDEX
        self.eij, self.ei = self.load_data(brawfile)

    def load_data(self, brawfile):
        if not brawfile.exists():
            raise FileNotFoundError(brawfile)
        data = msgpack.unpack(open(brawfile, 'rb'))
        L = self.seq_len
        V = len(self.vocab_index)
        eij = np.zeros((L, L, V, V))

        bytes_key = b'x_pair' in data.keys() # bug fix for MSG Pack

        if bytes_key:
            for i in range(L - 1):
                for j in range(i + 1, L):
                    arr = np.array(data[b'x_pair'][b'%d/%d' % (i, j)][b'x']).reshape(V, V)
                    eij[i, j] = arr
                    eij[j, i] = arr.T
            ei = np.array(data[b'x_single']).reshape(L, V - 1)
        else:
            for i in range(L - 1):
                for j in range(i + 1, L):
                    arr = np.array(data['x_pair']['%d/%d' % (i, j)]['x']).reshape(V, V)
                    eij[i, j] = arr
                    eij[j, i] = arr.T
            ei = np.array(data['x_single']).reshape(L, V - 1)
        return eij, ei

    def index_encoding(self, sequences, letter_to_index_dict):
        df = pd.DataFrame(iter(s) for s in sequences)
        encoding = df.replace(letter_to_index_dict)
        encoding = encoding.values.astype(np.int64)
        return encoding

    def ccmpred_encoding(self, index_encoded, profile='pair'):
        if profile == 'pair':
            encoding = all_sequence_pairwise_profile((self.eij, index_encoded))
        elif profile == 'single':
            encoding = all_sequence_singleton_profile((self.ei, index_encoded))
        else:
            raise NotImplementedError
        return encoding

    def encode(self, sequences):
        index_encoded = self.index_encoding(sequences, self.vocab_index)
        single = self.ccmpred_encoding(index_encoded, profile='single')
        pair = self.ccmpred_encoding(index_encoded, profile='pair')
        self.ccmpred_encoded = np.concatenate([single, pair], axis=2)
        return self.ccmpred_encoded
