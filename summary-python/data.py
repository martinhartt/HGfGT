# Load data for summary experiments.
import util
import torch
from torch.autograd import Variable


enable_cuda = True

class Data(object):
    """docstring for Data."""
    def __init__(self, title_data, article_data):
        super(Data, self).__init__()
        self.title_data = title_data
        self.article_data = article_data

    def reset(self):
        self.bucket_order = []
        for length, _ in self.title_data["target"].iteritems():
            self.bucket_order.append(length)

        util.shuffleTable(self.bucket_order) # Shuffle array
        self.bucket_index = 0
        self.load_next_bucket()

    def load_next_bucket(self):
        self.done_bucket = False
        self.bucket = self.bucket_order[self.bucket_index]
        self.bucket_size = self.title_data["target"][self.bucket].size(0)
        self.pos = 0
        self.aux_ptrs = self.title_data["sentences"][self.bucket].float().long() # ??
        self.positions = apply_cuda(torch.range(0, self.bucket - 1).view(1, self.bucket)
            .expand(1000, self.bucket).contiguous()) + (200 * self.bucket)
        self.bucket_index += 1

    def is_done(self):
        return self.bucket_index >= len(self.bucket_order) and self.done_bucket

    def next_batch(self, max_size):
        diff = self.bucket_size - self.pos
        if self.done_bucket or diff == 0 or diff == 1:
            self.load_next_bucket()

        if self.pos + max_size > self.bucket_order:
            offset = self.bucket_size - self.pos
            self.done_bucket = true
        else:
            offset = max_size

        positions = self.positions.narrow(0, 0, offset)

        try:
            temp = self.aux_ptrs.narrow(0, self.pos, offset)
            aux_rows = torch.index_select(self.article_data["words"][self.bucket], 0, temp)
            context = self.title_data["ngram"][self.bucket].narrow(0, self.pos, offset)
            target = self.title_data["target"][self.bucket].narrow(0, self.pos, offset)
            self.pos += offset
            return [Variable(aux_rows), Variable(positions), Variable(context)], target.long()
        except Exception as e:
            print('T2', '\nself.pos =', self.pos, '\noffset =', offset, '\nmax_size =', max_size, '\nself.bucket_order =', self.bucket_order, '\ndiff =', diff)
            self.done_bucket = True
            return self.next_batch(max_size)


def init(title_data, article_data):
    return Data(title_data, article_data)

def add_opts(parser):
   parser.add_argument('-articleDir', default='',
              help='Directory containing article training matrices.')
   parser.add_argument('-titleDir', default='',
              help='Directory containing title training matrices.')
   parser.add_argument('-validArticleDir', default='',
              help='Directory containing article matricess for validation.')
   parser.add_argument('-validTitleDir', default='',
              help='Directory containing title matrices for validation.')
   parser.add_argument('-cuda', default=False, type=bool,
              help='Enable cuda?')

def apply_cuda(tensor):
    return tensor.cuda() if enable_cuda else tensor

def load_title(dname, shuffle=None, use_dict=None):
    ngram = torch.load('{}ngram.mat.torch'.format(dname))
    words = torch.load('{}word.mat.torch'.format(dname))
    dict = use_dict or torch.load('{}dict'.format(dname))
    target_full = {}
    sentences_full = {}
    pos_full = {}

    for length, mat in ngram.iteritems():
        if shuffle != None:
            perm = torch.randperm(ngram[length].size(0))
            ngram[length] = apply_cuda(torch.index_select(ngram[length], 0, perm).float())
            words[length] = torch.index_select(words[length], 0, perm)
        else:
            ngram[length] = apply_cuda(ngram[length].float())
            assert(ngram[length].size(0) == words[length].size(0))

        target_full[length] = apply_cuda(words[length][:, 0].contiguous().float())
        sentences_full[length] = apply_cuda(words[length][:, 1].contiguous().float())
        pos_full[length] = words[length][:, 2]

        title_data = {"ngram": ngram,
                              "target": target_full,
                              "sentences": sentences_full,
                              "pos": pos_full,
                              "dict": dict}
    return title_data

def load_article(dname, use_dict=None):
    input_words = torch.load('{}word.mat.torch'.format(dname))
    # offsets = torch.load('{}offset.mat.torch'.format(dname))

    dict = use_dict or torch.load('{}dict'.format(dname))
    for length, mat in input_words.iteritems():
        input_words[length] = mat
        input_words[length] = apply_cuda(input_words[length].float())
        article_data = {"words": input_words, "dict": dict}
    return article_data
