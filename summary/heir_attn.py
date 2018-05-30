import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class HeirAttnEncoder(nn.Module):
    """docstring for LSTMEncoder."""
    def __init__(self, vocab_size, bow_dim, hidden_size, opt, glove_weights, K=7):
        super(HeirAttnEncoder, self).__init__()

        if opt.glove:
            self.summary_embedding = nn.Embedding.from_pretrained(glove_weights, freeze=False)
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

        last_hidden = hidden_summaries[-1] # last LSTM layer

        last_hidden = last_hidden.view(len(last_hidden), 1, -1)
        hidden_summaries, _ = self.control_lstm(last_hidden)

        hidden_summaries = hidden_summaries.view(self.K, self.hidden_size)

        return hij, hidden_summaries

class HeirAttnDecoder(nn.Module):
    """docstring for LSTMDecoder."""
    def __init__(self, vocab_size, bow_dim, hidden_size, opt, glove_weights=None, K=7):
        super(HeirAttnDecoder, self).__init__()

        self.K = K
        self.hidden_size = hidden_size
        self.bow_dim = bow_dim
        self.attention_dims = opt.attentionDims
        self.softmax_dims = self.hidden_size - self.attention_dims

        if opt.glove:
            self.context_embedding = nn.Embedding.from_pretrained(glove_weights, freeze=False)
        else:
            self.context_embedding = nn.Embedding(vocab_size, bow_dim)
        self.decoder_lstm = nn.LSTM(bow_dim, hidden_size)
        self.out_linear = nn.Linear(self.hidden_size if opt.noAttn else self.softmax_dims, vocab_size)

        self.opt = opt

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

        hij_collapsed = hij.view(batch_size, self.K * max_word_length, self.hidden_size)

        """
        Effective simple attention mechanism [Lopyrev, 2015] which splits the
        hidden vector into two sets: the first 472 dimensions for decoding
        words and the last 40 dimensions for computing the attention weight
        """
        decode_hij = hij_collapsed if self.opt.noAttn else hij_collapsed[:,:,:self.softmax_dims]

        if self.opt.noAttn:
            c = decode_hij
        else:
            attn_hij = hij_collapsed[:,:,-self.attention_dims:]
            attn_yh = y_h[:,-self.attention_dims:,:]
            attn_hidden_summaries = hidden_summaries[:,:,-self.attention_dims:]

            # Summary level attention
            a = torch.bmm(attn_hidden_summaries, attn_yh)
            a = F.softmax(a, dim=1)

            # Word level attention
            b = torch.bmm(attn_hij, attn_yh)
            b = b.view(batch_size, self.K, max_word_length)
            b = F.softmax(b, dim=2)

            # c = sum(ai * bij * hij)
            ab = torch.mul(a, b)
            ab = ab.view(batch_size, self.K * max_word_length, 1)

            c = torch.mul(ab, decode_hij)

        c = torch.sum(c, 1)

        out = self.out_linear(c)
        out = F.log_softmax(out, dim=1)

        return out
