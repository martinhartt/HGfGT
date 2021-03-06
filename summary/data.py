#  Copyright (c) 2018, Martin Hartt
#
#  Copyright (c) 2015, Facebook, Inc.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree. An additional grant
#  of patent rights can be found in the PATENTS file in the same directory.
#
#  Author: Alexander M Rush <srush@seas.harvard.edu>
#          Sumit Chopra <spchopra@fb.com>
#          Jason Weston <jase@fb.com>


# Load data for summary experiments.
import torch
import random
from torch.autograd import Variable
from util import apply_cuda, encode
from itertools import groupby

def add_opts(parser):
    parser.add_argument('--workingDir', default='')
    parser.add_argument('--hier', default=False, type=bool, help='Enable hierarchal model?')
    parser.add_argument('--maxSize', default=0.3 * (10 ** 5), type=bool, help='The maximum number of unextended samples per epoch')
    parser.add_argument(
        '--maxWordLength',
        type=int,
        default=52,
        help="maxWordLength.")
    parser.add_argument('--summLstmLayers', default=3, type=int, help='# of layers for a summary')
    parser.add_argument(
        '--K',
        type=int,
        default=7,
        help="Number of summaries to use.")

class BaseDataLoader(object):
    def __init__(self, input_file, dict, opt, window=5, max_size=0.3 * (10 ** 5)):
        super(BaseDataLoader, self).__init__()
        print("Using {} pairs per epoch".format(max_size))
        self.dict = dict
        self.input_file = input_file
        self.all_pairs = self.load_lazy(input_file, dict)
        self.max_size = max_size
        self.pairs = self.next_pairs(max_size)
        self.window = window
        self.opt = opt

    def next_pairs(self, max_size):
        i = 0

        new_pairs = []

        while i < max_size:
            try:
                new_pairs.append(next(self.all_pairs))
                i += 1
            except Exception as e:
                self.all_pairs = self.load_lazy(self.input_file, self.dict)
                break

        return new_pairs

    def reset(self):
        self.pairs = self.next_pairs(self.max_size)
        random.shuffle(self.pairs)

    def next_batch(self, max_batch_size):
        raise NotImplementedError()


    @staticmethod
    def load_lazy(input_file, dict):
        for line in open(input_file):
            if line.strip() == "":
                continue

            components = line.strip().split('\t')

            if len(components) not in [2,8]:
                continue

            title = components[0]
            articles = components[1:]

            title = encode(title, dict["w2i"])

            if len(articles) > 1:
                articles = [encode(article, dict["w2i"]) for article in articles]
            elif len(articles) > 0:
                articles = encode(articles[0], dict["w2i"])

            yield articles, title

    def get_every(self, iter, n):
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

class HierDataLoader(BaseDataLoader):
    """docstring for HierDataLoader."""

    def next_batch(self, max_batch_size):
        for pair in self.pairs:
            # When using the filtered dataset we must copy the lead sentence for summary
            if not isinstance(pair[0], (list, tuple)):
                pair = ([pair[0]] * self.opt.K, pair[1])

            expanded_group_iter = self.expand(pair, self.dict["w2i"], self.window)

            if any([summary.shape == () for summary in pair[0]]):
                continue

            article_summaries = self.torchify(pair[0], variable=True, revsort=True, opt=self.opt)

            # Skip if one of the length of summaries is zero
            if len([l for l in article_summaries[1] if l < 1]) != 0:
                continue

            for batch in self.get_every(expanded_group_iter, max_batch_size):
                batch = [pair for pair in batch if len(pair[0][1]) > 0]

                inputs, targets = zip(*batch)
                _, contexts = zip(*inputs)

                out = (article_summaries, self.torchify(contexts)), self.torchify(targets)
                yield out

    @staticmethod
    def torchify(arr, variable=False, revsort=False, opt=None):
        if variable:
            batch_size = len(arr)

            if revsort:
                arr = sorted(arr, key=len)[::-1]

            lengths = [len(batch) for batch in arr]

            largest_length = opt.maxWordLength

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
            # Return the whole context if using hierarchal attention model
            context = title[i:i + window]
            target = title[i + window]

            if len(context) < 1 or len(article) < 1:
                continue

            yield (article, context), target

    @staticmethod
    def make_input(article, context, K):
        a_tensors, a_lengths = article
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

            for batch in self.get_every(expanded_group_iter, max_batch_size):
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


def load_dict(filename):
    return torch.load(filename)
