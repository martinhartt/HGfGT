import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    """docstring for LSTMEncoder."""
    def __init__(self, vocab_size, bow_dim, hidden_size, K=7):
        super(LSTMEncoder, self).__init__()
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

class LSTMDecoder(nn.Module):
    """docstring for LSTMDecoder."""
    def __init__(self, vocab_size, bow_dim, hidden_size, K=7):
        super(LSTMDecoder, self).__init__()
        self.decoder_embedding = nn.Embedding(vocab_size, bow_dim)
        self.decoder_lstm = nn.LSTM(bow_dim, hidden_size)
        self.out_linear = nn.Linear(hidden_size, vocab_size)

        self.K = K
        self.hidden_size = hidden_size
        self.bow_dim = bow_dim

    def forward(self, title_ctx, encoder_out):
        hij, hidden_summaries = encoder_out

        padded_titles, title_lengths = title_ctx

        padded_titles = self.decoder_embedding(padded_titles)

        # The tensor is first packed due to variable lengths
        packed_titles = pack_padded_sequence(padded_titles, title_lengths, batch_first=True)

        _, (hidden_context, _) = self.decoder_lstm(packed_titles)
        y_h = hidden_context[-1]

        batch_size = len(title_lengths)

        y_h = y_h.view(batch_size, self.hidden_size, 1)

        a = torch.bmm(hidden_summaries, y_h)
        a = F.softmax(a, dim=1)
        a = a.view(batch_size, 1, self.K).repeat(1, self.K, 1)

        hij_collapsed = hij.view(batch_size, self.K * self.bow_dim, self.hidden_size)
        b = torch.bmm(hij_collapsed, y_h)
        b = b.view(batch_size, self.K, self.bow_dim)
        b = F.softmax(b, dim=2)

        ab = torch.bmm(a, b)

        ab = ab.view(batch_size, 1, self.K, self.bow_dim).repeat(1, self.hidden_size, 1, 1)
        ab = ab.view(batch_size, self.hidden_size, self.K *  self.bow_dim)

        hij = hij.view(batch_size, self.hidden_size, self.K *  self.bow_dim)

        c = torch.mul(ab, hij)
        c = torch.sum(c, 2)

        return self.out_linear(c)

input = torch.ones(7, 50).long()

for i in range(7):
    for j in range(50):
        input[i, j] = random.randint(1, 30)

input[5][47] = 0
input[5][48] = 0
input[5][49] = 0
input[6][46] = 0
input[6][47] = 0
input[6][48] = 0
input[6][49] = 0

input = (input, torch.tensor([50, 50, 50, 50, 50, 47, 46]))

BATCH_SIZE=32

def randArr(length):
    return [random.randint(1, 9999) for _ in range(length)]

HIDDEN_SIZE = 500
EMBEDDING_SIZE = 50

context = (torch.tensor([randArr(EMBEDDING_SIZE) for _ in range(BATCH_SIZE)]).long(), torch.arange(EMBEDDING_SIZE, EMBEDDING_SIZE-BATCH_SIZE, step=-1).long())
target = torch.tensor(randArr(BATCH_SIZE)).long()


encoder = LSTMEncoder(10000, EMBEDDING_SIZE, 500)
decoder = LSTMDecoder(10000, EMBEDDING_SIZE, 500)

encoder_out = encoder(input)
hij, hidden_summaries = encoder_out

encoder_out = hij.unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1), hidden_summaries.unsqueeze(0).repeat(BATCH_SIZE, 1, 1)

decoder_out = decoder(context, encoder_out)
