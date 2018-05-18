# Load data for summary experiments.
import torch
import random
from torch.autograd import Variable
from util import apply_cuda
from itertools import groupby

def add_opts(parser):
    parser.add_argument('-workingDir', default='')
    parser.add_argument('-cuda', default=False, type=bool, help='Enable cuda?')
    parser.add_argument('-heir', default=False, type=bool, help='Enable heirarchal model?')
    parser.add_argument('-small', default=False, type=bool, help='Use small data?')
    parser.add_argument(
        '-batchSize', type=int, default=64, help="Size of training minibatch.")

class BaseDataLoader(object):
    def __init__(self, title_data, article_data, dict, window=5):
        super(BaseDataLoader, self).__init__()
        self.pairs = zip(article_data, title_data)
        self.dict = dict
        self.window = window
        self.length_cache = None

        # Shuffle?
        self.reset()

    def reset(self):
        random.shuffle(self.pairs)

    def next_batch(self, max_batch_size):
        raise NotImplementedError()

    def expandIter(self, iter, w2i, window):
        for pair in iter:
            for expanded in self.expand(pair, w2i, window):
                yield expanded

    def getEvery(self, iter, n):
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


    def torchify(self, arr, variable=False):
        raise NotImplementedError()

    def expand(self, pair, w2i, window):
        raise NotImplementedError()

class HeirDataLoader(BaseDataLoader):
    """docstring for HeirDataLoader."""

    def next_batch(self, max_batch_size):
        for pair in self.pairs:
            expanded_group_iter = self.expand(pair, self.dict["w2i"], self.window)

            articles = self.torchify(pair[0], variable=True, revsort=True)

            for batch in self.getEvery(expanded_group_iter, max_batch_size):
                batch = batch[::-1]

                inputs, targets = zip(*batch)
                _, contexts = zip(*inputs)

                out = (articles, self.torchify(contexts, variable=True)), self.torchify(targets)
                yield out

    def sortInputs(self, input): # Sort in descending order
        return sorted(input, key=lambda a: len(a[0][1]))[::-1]

    def torchify(self, arr, variable=False, revsort=False):
        if variable:
            batch_size = len(arr)

            if revsort:
                arr = sorted(arr, key=len)[::-1]

            lengths = [len(batch) for batch in arr]
            largest_length = max(lengths)

            out = torch.zeros(batch_size, largest_length).long()

            # HACK There must be a better way to do this
            for batch in range(batch_size):
                for j in range(lengths[batch]):
                    out[batch][j] = arr[batch][j]

            return apply_cuda(Variable(out)), lengths
        else:
            return apply_cuda(Variable(torch.tensor(list(arr)).long()))


    def expand(self, pair, w2i, window):
        # Padding
        article, title = pair

        window = 1

        title = [w2i["<s>"]] * window + [int(x) for x in title] + [w2i["</s>"]]  # Append end of string
        article = [[w2i["<s>"]] + [int(x) for x in a] + [w2i["</s>"]] for a in article]

        for i in range(len(title) - window):
            # Return the whole context if using heirarchal attention model
            context = title[0:i + window]
            target = title[i + window]

            yield (article, context), target


class AbsDataLoader(BaseDataLoader):
    """docstring for AbsDataLoader."""
    def next_batch(self, max_batch_size):
        for key, group_iter in groupby(self.pairs, lambda x: len(x[0])):
            expanded_group_iter = self.expandIter(group_iter, self.dict["w2i"], self.window, self.heir)

            for batch in self.getEvery(expanded_group_iter, max_batch_size):
                inputs, targets = zip(*batch)
                articles, contexts = zip(*inputs)

                input = (self.torchify(articles), self.torchify(contexts))
                yield input, self.torchify(targets)


    def torchify(self, arr, variable=False):
        return apply_cuda(Variable(torch.tensor(list(arr)).long()))

    def expand(self, pair, w2i, window):
        # Padding
        article, title = pair

        title = [w2i["<s>"]] * window + [int(x) for x in title] + [w2i["</s>"]]  # Append end of string
        article = [w2i["<s>"]] * 3 + [int(x) for x in article] + [w2i["</s>"]] * 3


        for i in range(len(title) - window):
            # Return only a context of window size
            context = title[i:i + window]
            target = title[i + window]

            yield (article, context), target


def load(dname, train=True, type="dict", heir=True, small=True):
    prefix = "/all." if heir else "/filter."
    prefix += "small_" if small else ""
    prefix += "train" if train else "valid"

    return torch.load('{}{}.{}.torch'.format(dname, prefix, type))

def make_input(article, context, K):
    bucket = article.size(0)
    article_tensor = apply_cuda(article.view(bucket, 1)
        .expand(bucket, K)
        .t()
        .contiguous())

    return [Variable(tensor.long()) for tensor in [article_tensor, context]]
