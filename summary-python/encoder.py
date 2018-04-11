import torch
import torch.legacy.nn as tnn
import torch.nn as nn

enable_cuda = True


def add_opts(parser):
    parser.add_argument('-encoderModel', default='bow',help= "The encoder model to use.")
    parser.add_argument('-bowDim', type=int,      default=50, help="Article embedding size.")
    parser.add_argument('-attenPool', type=int,    default=5, help="Attention model pooling size.")
    parser.add_argument('-hiddenUnits', type=int, default=1000, help="Conv net encoder hidden units.")
    parser.add_argument('-kernelWidth', type=int, default=5,   help= "Conv net encoder kernel width.")

def build(opt, data):
    torch.setdefaulttensortype("torch.CudaTensor")
    model = None
    if opt.encoderModel == "none":
        model = encoder.build_blank_model(opt, data)
    elif opt.encoderModel == "bow":
        model =  encoder.build_bow_model(opt, data)
    elif opt.encoderModel == "attenbow":
        model = encoder.build_attnbow_model(opt, data)
    elif opt.encoderModel == "conv":
        model = encoder.build_conv_model(opt, data)

    torch.setdefaulttensortype("torch.DoubleTensor")
    return model


class AttnBowEncoder(nn.Module):
    """docstring for AttnBowEncoder."""
    def __init__(self, bow_dim, window_size, title_vocab_size, article_vocab_size, opt):
        super(AttnBowEncoder, self).__init__()

        self.bow_dim = bow_dim # D2
        self.window_size = window_size # N
        self.title_vocab_size = title_vocab_size # V
        self.article_vocab_size = article_vocab_size # V2
        self.opt = opt

        self.article_embedding = nn.Embedding(article_vocab_size, bow_dim)
        self.title_embedding = nn.Embedding(title_vocab_size, bow_dim)

        self.pad = (opt.attenPool - 1) / 2
        self.title_context = nn.Linear(window_size * bow_dim, bow_dim)

        self.non_linearity = nn.Softmax()

        self.mout_linear = nn.Linear(bow_dim, bow_dim)

    def forward(self, article, size, title):
        article = self.article_embedding(article.long())
        title = self.title_embedding(title.long())

        n = article.shape[0]

        # test = torch.matmul(article, title.t())
        title = title.view(n, self.window_size * self.bow_dim)
        # torch.save(title, 'TEST.pt')
        title = self.title_context(title)
        title = title.view(n, self.bow_dim, 1)


        dot_article_context = torch.matmul(article, title)

        attention = torch.sum(dot_article_context, 2)
        attention = self.non_linearity(attention)


        process_article = article.view(n, 1, -1, self.bow_dim)
        process_article = nn.ZeroPad2d((0, 0, self.pad, self.pad)).forward(process_article)
        print('DEBUG', self.opt.attenPool, process_article.shape)
        process_article = tnn.SpatialSubSampling(1, 1, self.opt.attenPool).forward(process_article)
        process_article = torch.sum(process_article, 1)

        attention = attention.view(n, -1, 1)
        process_article = torch.transpose(process_article, 1, 2)
        m = torch.matmul(process_article, attention)
        out = torch.sum(m, 2)
        out = self.mout_linear(out)
        return out
