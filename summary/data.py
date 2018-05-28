# Load data for summary experiments.
import torch
import random
from torch.autograd import Variable
from util import apply_cuda, encode
from itertools import groupby

def add_opts(parser):
    parser.add_argument('--workingDir', default='')
    parser.add_argument('--heir', default=False, type=bool, help='Enable heirarchal model?')
    parser.add_argument('--maxSize', default=10**6, type=bool, help='The maximum number of unextended samples per epoch')

class BaseDataLoader(object):
    def __init__(self, inputFile, dict, window=5, maxSize=10 ** 6):
        super(BaseDataLoader, self).__init__()
        self.dict = dict
        self.inputFile = inputFile
        self.allPairs = self.loadLazy(inputFile, dict)
        self.maxSize = maxSize
        self.pairs = self.next_pairs(maxSize)
        self.window = window

    def next_pairs(self, maxSize):
        i = 0

        newPairs = []

        while i < maxSize:
            try:
                newPairs.append(next(self.allPairs))
                i += 1
            except Exception as e:
                self.allPairs = self.loadLazy(self.inputFile, self.dict)
                break

        return newPairs

    def reset(self):
        self.pairs = self.next_pairs(self.maxSize)
        random.shuffle(self.pairs)

    def next_batch(self, max_batch_size):
        raise NotImplementedError()


    @staticmethod
    def loadLazy(inputFile, dict):
        for line in open(inputFile):
            if line.strip() == "":
                continue

            components = line.strip().split('\t')

            title = components[0]
            articles = components[1:]

            title = encode(title, dict["w2i"])

            if len(articles) > 1:
                articles = [encode(article, dict["w2i"]) for article in articles]
            elif len(articles) > 0:
                articles = encode(articles[0], dict["w2i"])

            yield articles, title

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

    @staticmethod
    def torchify(arr, variable=False):
        raise NotImplementedError()

    @staticmethod
    def expand(pair, w2i, window):
        raise NotImplementedError()

class HeirDataLoader(BaseDataLoader):
    """docstring for HeirDataLoader."""

    def next_batch(self, max_batch_size):
        for pair in self.pairs:
            expanded_group_iter = self.expand(pair, self.dict["w2i"], self.window)

            article_summaries = self.torchify(pair[0], variable=True, revsort=True)

            # Skip if one of the length of summaries is zero
            if len([l for l in article_summaries[1] if l < 1]) != 0:
                continue

            for batch in self.getEvery(expanded_group_iter, max_batch_size):
                batch = [pair for pair in batch if len(pair[0][1]) > 0]

                inputs, targets = zip(*batch)
                _, contexts = zip(*inputs)

                out = (article_summaries, self.torchify(contexts, variable=True)), self.torchify(targets)
                yield out

    def sortInputs(self, input): # Sort in descending order
        return sorted(input, key=lambda a: len(a[0][1]))[::-1]

    @staticmethod
    def torchify(arr, variable=False, revsort=False):
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

    @staticmethod
    def expand(pair, w2i, window):
        # Padding
        article, title = pair

        window = 1

        title = [w2i["<s>"]] * window + [int(x) for x in title] + [w2i["</s>"]]  # Append end of string
        article = [[w2i["<s>"]] + [int(x) for x in a] + [w2i["</s>"]] for a in article]

        for i in range(len(title) - window):
            # Return the whole context if using heirarchal attention model
            context = title[0:i + window]
            target = title[i + window]

            if len(context) < 1 or len(article) < 1:
                continue

            yield (article, context), target

    @staticmethod
    def make_input(article, context, K):
        a_tensors, a_lengths = article
        print(a_tensors)
        bucket = article.size(0)
        article_tensor = apply_cuda(article.view(bucket, 1)
            .expand(bucket, K)
            .t()
            .contiguous())

        return [Variable(tensor.long()) for tensor in [article_tensor, context]]



class AbsDataLoader(BaseDataLoader):
    """docstring for AbsDataLoader."""
    def next_batch(self, max_batch_size):
        for key, group_iter in groupby(self.pairs, lambda x: len(x[0])):
            expanded_group_iter = self.expandIter(group_iter, self.dict["w2i"], self.window)

            for batch in self.getEvery(expanded_group_iter, max_batch_size):
                inputs, targets = zip(*batch)
                articles, contexts = zip(*inputs)

                input = (self.torchify(articles), self.torchify(contexts))
                yield input, self.torchify(targets)


    def expandIter(self, iter, w2i, window):
        for pair in iter:
            for expanded in self.expand(pair, w2i, window):
                yield expanded

    @staticmethod
    def torchify(arr, variable=False):
        return apply_cuda(Variable(torch.tensor(list(arr)).long()))

    @staticmethod
    def expand(pair, w2i, window):
        # Padding
        article, title = pair

        title = [w2i["<s>"]] * window + [int(x) for x in title] + [w2i["</s>"]]  # Append end of string
        article = [w2i["<s>"]] * 3 + [int(x) for x in article] + [w2i["</s>"]] * 3


        for i in range(len(title) - window):
            # Return only a context of window size
            context = title[i:i + window]
            target = title[i + window]

            yield (article, context), target

    @staticmethod
    def make_input(article, context, K):
        bucket = article.size(0)
        article_tensor = apply_cuda(article.view(bucket, 1)
            .expand(bucket, K)
            .t()
            .contiguous())

        return [Variable(tensor.long()) for tensor in [article_tensor, context]]


def loadDict(filename):
    return torch.load(filename)
