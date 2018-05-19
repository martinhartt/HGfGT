import torch
import argparse
from util import apply_cuda
import data
import re
import math

from extractive import extractive

from data import BaseDataLoader, HeirDataLoader

parser = argparse.ArgumentParser(description='Train a summarization model.')

parser.add_argument('-modelFilename', default='', help='Model to test.')
parser.add_argument('-inputf', default='', help='Input article files. ')
parser.add_argument('-outputf', default='', help='Actual title files. ')
parser.add_argument(
    '-length', type=int, default=15, help='Maximum length of summary.')
parser.add_argument(
    '-allowUNK', type=bool, default=False, help="Allow generating <unk>.")
parser.add_argument(
    '-fixedLength',
    type=bool,
    default=True,
    help="Produce exactly -length words.")
parser.add_argument(
    '-blockRepeatWords',
    type=bool,
    default=False,
    help="Disallow generating a word twice.")
parser.add_argument(
    '-lmWeight', type=float, default=1.0, help="Weight for main model.")
parser.add_argument(
    '-beamSize', type=int, default=100, help="Size of the beam.")
parser.add_argument(
    '-showCandidates',
    type=bool,
    default=False,
    help="If true, shows next most likely summaries.")

data.add_opts(parser)

opt = parser.parse_args()
data.enable_cuda = opt.cuda

def normalize(sent):
    sent = sent.lower()
    sent = re.sub(r"([.!?])", r" \1", sent)
    sent = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", sent)
    sent = re.sub(r'\d', '#', sent)
    return sent

def process_word(input_word):
    return re.sub(r'\d', '#', input_word.lower())

def encode(sent, w2i):
    return [w2i.get(word, w2i["<unk>"]) for word in sent.split()]

def main():
    state = torch.load(opt.modelFilename)

    print("Heir is {}".format(opt.heir))
    if opt.heir:
        mlp, encoder = state
    else:
        mlp = state

    dict = data.load(opt.workingDir, train=True, type='dict', heir=opt.heir, small=opt.small)

    sent_file = open(opt.inputf).read().split("\n")
    length = opt.length
    if not opt.heir:
        W = mlp.window
        opt.window = mlp.window
    else:
        W = 1

    w2i = dict["w2i"]
    i2w = dict["i2w"]

    K = opt.beamSize
    FINAL_VAL = 1000
    INF = float('inf')

    UNK = w2i["<unk>"]
    START = w2i["<s>"]
    END = w2i["</s>"]


    actual = open(opt.outputf).read().split('\n')

    sent_num = 0
    for line in sent_file:
        # Add padding
        if opt.heir:
            summaries = extractive(line).split("\t")
            encoded_summaries = [encode(normalize(summary), w2i) for summary in summaries]
            article = HeirDataLoader.torchify(encoded_summaries, variable=True, revsort=True)
        else:
            true_line = "<s> <s> <s> {} </s> </s> </s>".format(normalize(line))

            article = torch.tensor(encode(true_line, w2i))

        n = opt.length

        # Initilize the charts.
        # scores[i][k] is the log prob of the k'th hyp of i words.
        # hyps[i][k] contains the words in k'th hyp at
        #          i word (left padded with W <s>) tokens.
        result = []
        scores = apply_cuda(torch.zeros(n + 1, K).float())
        hyps = apply_cuda(torch.zeros(n + 1, K, W + n + 1).long().fill_(START))

        for i in range(n):
            # For each output word
            context = hyps[i].narrow(1, 0 if opt.heir else i + 1, i+1 if opt.heir else W)
            cur_K = K

            # (1) Score all next words for each context in the beam.
            #    log p(y_{i+1} | y_c, x) for all y_c

            if opt.heir:
                encoder_out = encoder(article)
                model_scores = mlp(encoder_out, (context, [i+1] * K))
            else:
                input = DataLoader.make_input(article, context, cur_K)
                model_scores = mlp(*input)

            out_scores = model_scores.data.clone().mul(opt.lmWeight)

            # If length limit is reached, next word must be end.
            finalized = (i == n - 1) and opt.fixedLength

            if finalized:
                out_scores[:, END] += FINAL_VAL
            else:
                # Apply hard constraints
                out_scores[:, START] = -INF
                if not opt.allowUNK:
                    out_scores[:, UNK] = -INF

                if opt.fixedLength:
                    out_scores[:, END] = -INF

            # Only take first row when starting out.
            if i == 0:
                cur_K = 1

            # Prob of summary is log p + log p(y_{i+1} | y_c, x)
            for k in range(cur_K):
                out_scores[k] += scores[i][k]

            # (2) Retain the K-best words for each hypothesis using GPU.
            # This leaves a KxK matrix which we flatten to a K^2 vector.
            max_scores, mat_indices = torch.topk(apply_cuda(out_scores), K)

            flat = max_scores.view(
                max_scores.size(0) * max_scores.size(1)).float()

            # 3) Construct the next hypotheses by taking the next k-best.
            for k in range(K):
                for _ in range(K):
                    # (3a) Pull the score, index, rank, and word of the
                    # current best in the table, and then zero it out.
                    score, index = torch.max(flat, 0)
                    score = float(score)
                    index = int(index)

                    if finalized:
                        score -= FINAL_VAL

                    scores[i + 1][k] = score # For the next word

                    def flat_to_rc(v, indices, flat_index):
                        row = int(math.floor((flat_index) / v.size(1)))
                        col = int((flat_index) % (v.size(1)))
                        return row, indices[row][col]

                    prev_k, y_i1 = flat_to_rc(max_scores, mat_indices, index)

                    flat[index] = -INF

                    # (3c) Add the word, its score, and its features to the
                    # beam.
                    # Update tables with new hypothesis
                    for j in range(i + W):
                        hyps[i + 1][k][j] = hyps[i][prev_k][j]

                    hyps[i + 1][k][i + W] = y_i1

                    # If we have produced an END symbol, push to stack
                    if y_i1 == END:
                        result.append((i + 1, scores[i + 1][k],
                                       hyps[i + 1][k].clone()))
                        scores[i + 1][k] = -INF


        sorted_results = sorted(result, key=lambda a: a[1])

        for (rank, (_, score,
                    output)) in enumerate(sorted_results[:1]):

            final = ""

            for j in range(W + 2, W + length - 1):
                index = int(output[j])
                word = i2w[index]
                final += " {}".format(word)

            print("")
            print("> {}".format(line))
            print("= {}".format(actual[sent_num]))
            print("< {}".format(final.strip()))
            print("")

        sent_num += 1



main()
