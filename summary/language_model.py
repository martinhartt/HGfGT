import torch
import torch.nn as nn
import encoder
from glove import build_glove

class LanguageModel(nn.Module):
    """docstring for LanguageModel."""

    def __init__(self, dict, opt):
        super(LanguageModel, self).__init__()
        self.window = opt.window  # N
        self.hidden_size = opt.hiddenSize  # H
        self.vocab_size = len(dict["i2w"])  # V

        self.bow_dim = opt.bowDim
        if opt.glove:
            glove_weights = build_glove(dict["w2i"])
            self.context_embedding = nn.Embedding.from_pretrained(glove_weights, freeze=True)
        else:
            self.context_embedding = nn.Embedding(self.vocab_size, self.bow_dim)
            glove_weights = None

        self.encoder = encoder.AttnBowEncoder(self.bow_dim, self.window, self.vocab_size, opt, glove_weights)

        self.context_linear = nn.Linear(self.bow_dim * self.window,
                                        self.hidden_size)
        self.context_tanh = nn.Tanh()
        self.out_linear = nn.Linear(self.hidden_size + self.bow_dim,
                                    self.vocab_size)
        self.soft_max = nn.LogSoftmax()

    def forward(self, article, title_ctx):
        batch_size = article.shape[0]

        article = self.encoder(article, title_ctx)

        title_ctx = self.context_embedding(title_ctx)

        # tanh W (E y)
        title_ctx = title_ctx.view(batch_size, self.bow_dim * self.window)
        title_ctx = self.context_linear(title_ctx)
        title_ctx = self.context_tanh(title_ctx)

        # Second layer: takes LM and encoder model.
        out = torch.cat((title_ctx, article), 1)
        out = out.view(batch_size, self.hidden_size + self.bow_dim)
        out = self.out_linear(out)
        out = self.soft_max(out)
        return out
