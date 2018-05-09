import torch
import torch.nn as nn

import nnlm
import argparse
from util import apply_cuda
import data
import re
import math

parser = argparse.ArgumentParser(description='Train a summarization model.')

parser.add_argument('-modelFilename', default='', help='Model to test.')
parser.add_argument('-inputf', default='', help='Input article files. ')
parser.add_argument(
    '-nbest',
    type=bool,
    default=False,
    help='Write out the nbest list in ZMert format.')
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
    '-extractive',
    type=bool,
    default=False,
    help="Force fully extractive summary.")
parser.add_argument(
    '-abstractive',
    type=bool,
    default=False,
    help="Force fully abstractive summary.")
parser.add_argument(
    '-recombine',
    type=bool,
    default=False,
    help="Used hypothesis recombination.")
parser.add_argument(
    '-showCandidates',
    type=bool,
    default=False,
    help="If true, shows next most likely summaries.")

data.add_opts(parser)

opt = parser.parse_args()
data.enable_cuda = opt.cuda


# Map the words from one dictionary to another.
def sync_dicts(dict1, dict2):
    dict_map = torch.ones(len(dict1["i2w"])).long()
    for i in range(len(dict1["i2w"])):
        temp = dict1["i2w"][i]
        try:
            res = dict2["w2i"][temp]
            dict_map[i] = res or 0
        except Exception as e:
            dict_map[i] = 0

    return dict_map


def process_word(input_word):
    return re.sub(r'\d', '#', input_word.lower())


def main():
    mlp = torch.load(opt.modelFilename)

    article_data = data.load_article(opt.articleDir)
    adict = article_data["dict"]

    tdata = data.load_title(opt.titleDir, True)
    tdict = tdata["dict"]

    dict_map = sync_dicts(adict, tdict)
    sent_file = open(opt.inputf).read().split("\n")
    length = opt.length
    W = mlp.window
    a_w2i = adict["w2i"]
    a_i2w = adict["i2w"]
    t_w2i = tdict["w2i"]
    t_i2w = tdict["i2w"]

    K = opt.beamSize
    FINAL_VAL = 1000
    INF = float('inf')

    UNK = t_w2i["<unk>"]
    START = t_w2i["<s>"]
    END = t_w2i["</s>"]

    W = mlp.window
    opt.window = mlp.window

    sent_num = 0
    for line in sent_file:
        if opt.showCandidates:
            print("\n{}\n============".format(line))

        sent_num += 1

        # Add padding
        true_line = "<s> <s> <s> {} </s> </s> </s>".format(line)
        words = true_line.split()

        article = torch.zeros(len(words))
        for j in range(0, len(words)):
            word = process_word(words[j])
            try:
                article[j] = a_w2i[word] or a_w2i["<unk>"]
            except Exception:
                article[j] = a_w2i["<unk>"]

        n = opt.length

        # Initilize the charts.
        # scores[i][k] is the log prob of the k'th hyp of i words.
        # hyps[i][k] contains the words in k'th hyp at
        #          i word (left padded with W <s>) tokens.
        result = []
        scores = apply_cuda(torch.zeros(n + 1, K).float())
        hyps = apply_cuda(torch.zeros(n + 1, K, W + n + 1).long().fill_(START))

        for i in range(n):
            cur_beam = hyps[i].narrow(1, i + 1, W)
            cur_K = K

            # (1) Score all next words for each context in the beam.
            #    log p(y_{i+1} | y_c, x) for all y_c
            input = data.make_input(article, cur_beam, cur_K)
            model_scores = mlp(*input)
            out_scores = model_scores.data.clone().double().mul(opt.lmWeight)

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
                # out_scores = out_scores.narrow(1, 1, -1)
                model_scores = model_scores.narrow(1, 1, 1)

            # Prob of summary is log p + log p(y_{i+1} | y_c, x)
            for k in range(cur_K):
                out_scores[k] = out_scores[k] + scores[i][k]

            # (2) Retain the K-best words for each hypothesis using GPU.
            # This leaves a KxK matrix which we flatten to a K^2 vector.
            max_scores, mat_indices = torch.topk(apply_cuda(out_scores), K)
            flat = max_scores.view(
                max_scores.size(0) * max_scores.size(1)).float()

            # 3) Construct the next hypotheses by taking the next k-best.
            for k in range(K):
                for _ in range(100):
                    # (3a) Pull the score, index, rank, and word of the
                    # current best in the table, and then zero it out.
                    score, index = torch.max(flat, 0)
                    score = float(score)
                    index = int(index)

                    if finalized:
                        score -= FINAL_VAL

                    scores[i + 1][k] = score

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

        numOfCandidates = 5 if opt.showCandidates else 1

        for (rank, (_, score,
                    output)) in enumerate(sorted_results[:numOfCandidates]):

            final = "\n{}.".format(rank + 1) if opt.showCandidates else ""

            for j in range(W + 2, W + length - 1):
                index = int(output[j])
                word = t_i2w[index]
                final += " {}".format(word)

            print(final.strip())


main()
