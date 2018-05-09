import torch
import torch.legacy.nn as tnn
import torch.nn as nn


def add_opts(parser):
    parser.add_argument(
        '-encoderModel', default='bow', help="The encoder model to use.")
    parser.add_argument(
        '-bowDim', type=int, default=50, help="Article embedding size.")
    parser.add_argument(
        '-attenPool',
        type=int,
        default=5,
        help="Attention model pooling size.")


class AttnBowEncoder(nn.Module):
    """docstring for AttnBowEncoder."""

    def __init__(self, bow_dim, window_size, vocab_size):
        super(AttnBowEncoder, self).__init__()

        self.bow_dim = bow_dim  # D2
        self.window_size = window_size  # N
        self.vocab_size = vocab_size  # V

        self.article_embedding = nn.Embedding(vocab_size, bow_dim)
        self.title_embedding = nn.Embedding(vocab_size, bow_dim)

        self.pad = (opt.attenPool - 1) / 2
        self.title_linear = nn.Linear(window_size * bow_dim, bow_dim)

        self.non_linearity = nn.Softmax()

        self.mout_linear = nn.Linear(bow_dim, bow_dim)
        self.pool_layer = nn.AvgPool2d((5, 1), stride=(1, 1))

    def forward(self, article, title_ctx):
        batch_size = article.shape[0]

        article = self.article_embedding(article.long())
        title_ctx = self.title_embedding(title_ctx.long())

        title_ctx = title_ctx.view(batch_size, self.window_size * self.bow_dim)
        title_ctx = self.title_linear(title_ctx)
        title_ctx = title_ctx.view(batch_size, self.bow_dim, 1)

        dot_article_context = torch.matmul(article, title_ctx)

        attention = torch.sum(dot_article_context, 2)  # ?
        attention = self.non_linearity(attention)
        attention = attention.view(batch_size, -1, 1)

        process_article = article.view(batch_size, 1, -1, self.bow_dim)
        process_article = nn.ZeroPad2d((0, 0, self.pad,
                                        self.pad)).forward(process_article)
        process_article = self.pool_layer.forward(process_article)
        process_article = torch.sum(process_article, 1)
        process_article = torch.transpose(process_article, 1, 2)

        m = torch.matmul(process_article, attention)
        out = torch.sum(m, 2)
        out = self.mout_linear(out)
        return out
