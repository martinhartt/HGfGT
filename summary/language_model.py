import torch
import torch.nn as nn
from util import apply_cuda

class LanguageModel(nn.Module):
    """docstring for LanguageModel."""
    def __init__(self, encoder, encoder_size, dictionary, opt):
        super(LanguageModel, self).__init__()
        self.embedding_dim = opt.embeddingDim # D
        self.window = opt.window # N
        self.hidden_size = opt.hiddenSize # H
        self.vocab_size = len(dictionary["index_to_symbol"]) # V
        self.encoder_size = encoder_size #P
        self.encoder = encoder

        self.context_lookup = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.context_linear = nn.Linear(self.embedding_dim * self.window, self.hidden_size)
        self.context_tanh = nn.Tanh()
        self.out_linear = nn.Linear(self.hidden_size + self.encoder_size, self.vocab_size)
        self.soft_max = nn.LogSoftmax()

    def forward(self, encoder_input, position_input, context_input): # context, encoder_input, position_input):
        context = self.context_lookup(context_input.long())
        encoder_input = self.encoder(encoder_input, position_input, context_input)

        n = context_input.shape[0]
        # tanh W (E y)
        context = context.view(n, self.embedding_dim * self.window)
        context = self.context_linear(context)
        context = self.context_tanh(context)

        # Second layer: takes LM and encoder model.
        out = torch.cat((context, encoder_input), 1)
        out = out.view(n, self.hidden_size + self.encoder_size)
        out = self.out_linear(out)
        out = self.soft_max(out)
        return out
