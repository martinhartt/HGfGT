# Load data for summary experiments.
import util
import torch
import random
from torch.autograd import Variable
from util import apply_cuda
from itertools import groupby


def add_opts(parser):
    parser.add_argument('-workingDir', default='')
    parser.add_argument('-filter', type=bool, default=True)
    parser.add_argument('-cuda', default=False, type=bool, help='Enable cuda?')
    parser.add_argument(
        '-batchSize', type=int, default=64, help="Size of training minibatch.")


class Data(object):
    """docstring for Data."""

    def __init__(self, title_data, article_data, data):
        super(Data, self).__init__()
        self.dict = dict
        self.pairs = zip(article_data, title_data)

        self.reset()

    def reset(self):
        random.shuffle(self.pairs)

    def next_batch(self, max_batch_size):
        for key, group in groupby(self.pairs, lambda x: len(x[0])):
            num_of_batches = len(group) / max_batch_size

            for i in range(num_of_batches):
                start = i * max_batch_size
                end = math.min(len(self.pairs), start + max_batch_size)

                yield self.pairs[start:end]


def load(dname, train=True, type="dict", filter=True):
    prefix = ".filter" if filter else ".all"
    prefix += ".train" if train else ".valid"
    return torch.load('{}{}.{}.torch'.format(dname, prefix, type))
