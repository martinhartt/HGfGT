import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class HeirAttnEncoder(nn.Module):
    """docstring for LSTMEncoder."""
    def __init__(self, vocab_size, bow_dim, hidden_size, opt, glove_weights, K=7):
        super(HeirAttnEncoder, self).__init__()

        if opt.glove:
            self.summary_embedding = nn.Embedding.from_pretrained(glove_weights, freeze=True)
        else:
            self.summary_embedding = nn.Embedding(vocab_size, bow_dim)

        self.summaries_lstm = nn.LSTM(bow_dim, hidden_size, 3)
        self.control_lstm = nn.LSTM(hidden_size, hidden_size, 1)
        self.hidden_size = hidden_size

        self.K = K

    def forward(self, summaries):
        padded_summaries, summary_lengths = summaries
        padded_summaries = self.summary_embedding(padded_summaries)

        # The tensor is first packed due to variable lengths
        packed = pack_padded_sequence(padded_summaries, summary_lengths, batch_first=True)

        sum_hidden_packed, (hidden_summaries, _) = self.summaries_lstm(packed)

        hij, _ = pad_packed_sequence(sum_hidden_packed, batch_first=True)

        last_hidden = hidden_summaries[-1]

        last_hidden = last_hidden.view(len(last_hidden), 1, -1)
        hidden_summaries, _ = self.control_lstm(last_hidden)

        hidden_summaries = hidden_summaries.view(self.K, self.hidden_size)

        return hij, hidden_summaries

class HeirAttnDecoder(nn.Module):
    """docstring for LSTMDecoder."""
    def __init__(self, vocab_size, bow_dim, hidden_size, opt, glove_weights=None, K=7):
        super(HeirAttnDecoder, self).__init__()

        if opt.glove:
            self.context_embedding = nn.Embedding.from_pretrained(glove_weights, freeze=True)
        else:
            self.context_embedding = nn.Embedding(vocab_size, bow_dim)
        self.decoder_lstm = nn.LSTM(bow_dim, hidden_size)
        self.out_linear = nn.Linear(hidden_size, vocab_size)

        self.K = K
        self.hidden_size = hidden_size
        self.bow_dim = bow_dim

    def forward(self, encoder_out, title_ctx):
        padded_titles, title_lengths = title_ctx
        batch_size = len(title_lengths)

        hij, hidden_summaries = encoder_out
        hij, hidden_summaries = hij.unsqueeze(0).repeat(batch_size, 1, 1, 1), hidden_summaries.unsqueeze(0).repeat(batch_size, 1, 1)
        max_word_length = hij.shape[2]


        padded_titles = self.context_embedding(padded_titles)

        # The tensor is first packed due to variable lengths
        packed_titles = pack_padded_sequence(padded_titles, title_lengths, batch_first=True)

        _, (hidden_context, _) = self.decoder_lstm(packed_titles)
        y_h = hidden_context[-1]


        y_h = y_h.view(batch_size, self.hidden_size, 1)

        # Summary level attention
        a = torch.bmm(hidden_summaries, y_h)
        a = F.softmax(a, dim=1)

        hij_collapsed = hij.view(batch_size, self.K * max_word_length, self.hidden_size)

        # Word level attention
        b = torch.bmm(hij_collapsed, y_h)
        b = b.view(batch_size, self.K, max_word_length)
        b = F.softmax(b, dim=2)

        # c = sum(ai * bij * hij)
        ab = torch.mul(a, b)
        ab = ab.view(batch_size, self.K * max_word_length, 1)

        c = torch.mul(ab, hij_collapsed)
        c = torch.sum(c, 1)

        out = self.out_linear(c)
        out = F.log_softmax(out, dim=1)

        return out
