import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from util import apply_cuda

class HeirAttnEncoder(nn.Module):
    """docstring for LSTMEncoder."""
    def __init__(self, vocab_size, bow_dim, hidden_size, opt, glove_weights):
        super(HeirAttnEncoder, self).__init__()

        if opt.glove:
            self.summary_embedding = nn.Embedding.from_pretrained(glove_weights, freeze=False)
        else:
            self.summary_embedding = nn.Embedding(vocab_size, bow_dim)

        self.summaries_lstm = nn.LSTM(bow_dim, hidden_size, opt.summLstmLayers)
        self.control_lstm = nn.LSTM(hidden_size, hidden_size, 1)
        self.hidden_size = hidden_size

        if opt.simple:
            self.simple_lstm = LSTM(bow_dim, hidden_size, 1)

        self.K = opt.K
        self.opt = opt

    def forward(self, summaries, hidden_state, summ_hidden_state):
        padded_summaries, summary_lengths = summaries
        padded_summaries = self.summary_embedding(padded_summaries)

        # The tensor is first packed due to variable lengths
        packed = pack_padded_sequence(padded_summaries, summary_lengths, batch_first=True)

        sum_hidden_packed, (hidden_summaries, hidden_cell) = self.summaries_lstm(packed, summ_hidden_state)
        summ_hidden_state = (hidden_summaries, hidden_cell)

        hij, _ = pad_packed_sequence(sum_hidden_packed, batch_first=True)

        padBy = self.opt.maxWordLength - hij.size(1)
        hij = F.pad(hij, (0, 0, 0, padBy))

        last_hidden = hidden_summaries[-1] # last LSTM layer

        last_hidden = last_hidden.view(len(last_hidden), 1, -1)
        hidden_summaries, hidden_state = self.control_lstm(last_hidden, hidden_state)

        hidden_summaries = hidden_summaries.view(self.K, self.hidden_size)

        return (hij, hidden_summaries), hidden_state, summ_hidden_state


    def init_hidden(self, n=1, K=1):
        return (apply_cuda(torch.zeros(n, K, self.hidden_size)), apply_cuda(torch.zeros(n, K, self.hidden_size)))

class HeirAttnDecoder(nn.Module):
    """docstring for LSTMDecoder."""
    def __init__(self, vocab_size, bow_dim, hidden_size, opt, glove_weights=None):
        super(HeirAttnDecoder, self).__init__()

        self.K = opt.K
        self.hidden_size = hidden_size
        self.bow_dim = bow_dim
        self.attention_dims = opt.attentionDims
        self.softmax_dims = self.hidden_size - self.attention_dims

        if opt.glove:
            self.context_embedding = nn.Embedding.from_pretrained(glove_weights, freeze=False)
        else:
            self.context_embedding = nn.Embedding(vocab_size, bow_dim)
        self.decoder_lstm = nn.LSTM(bow_dim, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_linear = nn.Linear(self.softmax_dims, vocab_size)

        self.max_word_length = opt.maxWordLength
        self.attn = nn.Linear(self.K * self.max_word_length, self.K * self.max_word_length)

        self.opt = opt

    def forward(self, encoder_out, context, lstm_hidden):
        batch_size = len(context)

        # hij is the hidden state for each word in each summary
        # hidden_summaries is the hidden state of every summary
        hij, hidden_summaries = encoder_out
        hij, hidden_summaries = hij.unsqueeze(0).repeat(batch_size, 1, 1, 1), hidden_summaries.unsqueeze(0).repeat(batch_size, 1, 1)

        context = self.context_embedding(context)
        context = self.dropout(context)

        context = context.view(1, batch_size, self.bow_dim)

        _, (hidden_context, cell_context) = self.decoder_lstm(context, lstm_hidden)

        y_h = hidden_context[-1]

        y_h = y_h.view(batch_size, self.hidden_size, 1)

        hij_collapsed = hij.view(batch_size, self.K * self.max_word_length, self.hidden_size)

        """
        Effective simple attention mechanism [Lopyrev, 2015] which splits the
        hidden vector into two sets: the first 472 dimensions for decoding
        words and the last 40 dimensions for computing the attention weight
        """
        decode_hij = hij_collapsed[:,:,:self.softmax_dims]

        attn_hij = hij_collapsed[:,:,-self.attention_dims:]
        attn_yh = y_h[:,-self.attention_dims:,:]
        attn_hidden_summaries = hidden_summaries[:,:,-self.attention_dims:]


        # Summary level attention
        a = torch.bmm(attn_hidden_summaries, attn_yh)
        a = F.softmax(a, dim=1)

        # Word level attention
        b = torch.bmm(attn_hij, attn_yh)
        b = b.view(batch_size, self.K, self.max_word_length)
        b = F.softmax(b, dim=2)

        # c = sum(ai * bij * hij)
        ab = torch.mul(a, b)
        if self.opt.extraAttnLinear:
            ab = ab.view(batch_size, self.K * self.max_word_length)
            ab = self.attn(ab)
            ab = ab.view(batch_size, self.K * self.max_word_length, 1)
        else:
            ab = ab.view(batch_size, self.K * self.max_word_length, 1)

        c = torch.mul(ab, decode_hij)
        c = torch.sum(c, 1)

        out = self.out_linear(c)
        out = F.log_softmax(out, dim=1)

        return out, (hidden_context, cell_context), ab
