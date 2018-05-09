# Load data for summary experiments.
import torch
import random
from torch.autograd import Variable
from util import apply_cuda
from itertools import groupby
from tqdm import tqdm


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
        pairs = zip(article_data, title_data)

        self.inputs = [] # Expanded pairs

        for pair in tqdm(pairs):
            expanded = list(expand(pair, dict["w2i"], window))
            self.inputs.extend(expanded)

        # Shuffle?
        # self.reset()

    def reset(self):
        random.shuffle(self.inputs)

    def next_batch(self, max_batch_size):
        for key, group_iter in groupby(self.inputs, lambda x: len(x[0][0])):
            group = list(group_iter)
            num_of_batches = 1 + len(group) / max_batch_size

            for i in range(num_of_batches):
                start = i * max_batch_size
                end = min(len(group), start + max_batch_size)

                inputs, targets = zip(*group[start:end])
                articles, contexts = zip(*inputs)
                # TODO Wrap in variable

                input = (torchify(articles), torchify(contexts))
                yield input, torchify(targets)

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
