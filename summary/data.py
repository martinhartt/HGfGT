# Load data for summary experiments.
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

    def __init__(self, title_data, article_data, dict, window=5):
        super(Data, self).__init__()
        self.pairs = zip(article_data, title_data)
        self.dict = dict
        self.window = window

        # Shuffle?
        # self.reset()

    def reset(self):
        random.shuffle(self.pairs)

    def next_batch(self, max_batch_size):
        for key, group_iter in groupby(self.pairs, lambda x: len(x[0])):
            expanded_group_iter = expandIter(group_iter, self.dict["w2i"], self.window)

            for batch in getEvery(expanded_group_iter, max_batch_size):
                inputs, targets = zip(*batch)
                articles, contexts = zip(*inputs)

                input = (torchify(articles), torchify(contexts))
                yield input, torchify(targets)

def expandIter(iter, w2i, window):
    for pair in iter:
        for expanded in expand(pair, w2i, window):
            yield expanded

def getEvery(iter, n):
    i = 0
    arr = []
    for item in iter:
        arr.append(item)
        i += 1
        if i % n == 0:
            yield arr
            arr = []
    if len(arr) > 0:
        yield arr


def torchify(arr):
    return apply_cuda(Variable(torch.tensor(list(arr)).long()))

def expand(pair, w2i, window):
    # Padding
    article, title = pair

    title = [w2i["<s>"]] * window + [int(x) for x in title] + [
        w2i["</s>"]
    ]  # Append end of string
    article = [w2i["<s>"]] * 3 + [int(x) for x in article
                                       ] + [w2i["</s>"]] * 3

    for i in range(len(title) - window):
        article_tensor = article
        context_tensor = title[i:i + window]
        target_tensor = title[i + window]

        yield (article_tensor, context_tensor), target_tensor

def load(dname, train=True, type="dict", filter=True):
    prefix = "/filter" if filter else "/all"
    prefix += ".train" if train else ".valid"
    return torch.load('{}{}.{}.torch'.format(dname, prefix, type))

def make_input(article, context, K):
    bucket = article.size(0)
    article_tensor = apply_cuda(article.view(bucket, 1)
        .expand(bucket, K)
        .t()
        .contiguous())

    return [Variable(tensor.long()) for tensor in [article_tensor, context]]
