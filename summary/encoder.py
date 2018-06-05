import torch
import torch.nn as nn
from glove import build_glove


class AttnBowEncoder(nn.Module):
    """docstring for AttnBowEncoder."""

    def __init__(self, bow_dim, window, vocab_size, opt, glove_weights=None):
        super(AttnBowEncoder, self).__init__()

        self.bow_dim = bow_dim  # D2
        self.window = window  # N
        self.vocab_size = vocab_size  # V
        self.pooling = opt.pooling

        if opt.glove:
            self.article_embedding = nn.Embedding.from_pretrained(glove_weights, freeze=False)
            self.context_embedding = nn.Embedding.from_pretrained(glove_weights, freeze=False)
        else:
            self.article_embedding = nn.Embedding(vocab_size, bow_dim)
            self.context_embedding = nn.Embedding(vocab_size, bow_dim)

        self.pad = (opt.attenPool - 1) / 2
        self.context_linear = nn.Linear(window * bow_dim, bow_dim)

        self.non_linearity = nn.Softmax()

        self.mout_linear = nn.Linear(bow_dim, bow_dim)

    def forward(self, article, title_ctx):
        batch_size = article.shape[0]


        article = self.article_embedding(article)

        title_ctx = self.context_embedding(title_ctx)

        title_ctx = title_ctx.view(batch_size, self.window * self.bow_dim)
        title_ctx = self.context_linear(title_ctx)
        title_ctx = title_ctx.view(batch_size, self.bow_dim, 1)

        dot_article_context = torch.matmul(article, title_ctx)

        attention = torch.sum(dot_article_context, 2)
        attention = self.non_linearity(attention)
        attention = attention.view(batch_size, -1, 1)

        if self.pooling:
            process_article = article.view(batch_size, 1, -1, self.bow_dim)
            process_article = nn.ZeroPad2d((0, 0, self.pad, self.pad)).forward(process_article)
            process_article = nn.AvgPool2d((5, 1), stride=(1, 1)).forward(process_article)
            article = torch.sum(process_article, 1)

        process_article = torch.transpose(article, 1, 2)
        m = torch.matmul(process_article, attention)
        out = torch.sum(m, 2)
        out = self.mout_linear(out)

        return out, attention
