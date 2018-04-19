import torch
import torch.nn as nn

import nnlm
import encoder
# import beam_search
import argparse
import util
from util import apply_cuda
import data
import re
import math

parser = argparse.ArgumentParser(description='Train a summarization model.')

# beam.addOpts(parser)

parser.add_argument('-modelFilename',  default='', help='Model to test.')
parser.add_argument('-inputf',         default='', help='Input article files. ')
parser.add_argument('-nbest',type=bool,       default=False, help='Write out the nbest list in ZMert format.')
parser.add_argument('-length',type=int,          default=15, help='Maximum length of summary.')
parser.add_argument('-allowUNK', type=bool,          default=False, help="Allow generating <unk>.")
parser.add_argument('-fixedLength', type=bool,       default=True,  help="Produce exactly -length words.")
parser.add_argument('-blockRepeatWords', type=bool,  default=False, help="Disallow generating a word twice.")
parser.add_argument('-lmWeight', type=float,            default=1.0, help="Weight for main model.")
parser.add_argument('-beamSize', type=int,            default=100, help="Size of the beam.")
parser.add_argument('-extractive', type=bool,        default=False, help="Force fully extractive summary.")
parser.add_argument('-abstractive', type=bool,       default=False, help="Force fully abstractive summary.")
parser.add_argument('-recombine', type=bool,         default=False, help="Used hypothesis recombination.")


data.add_opts(parser)

opt = parser.parse_args()
data.enable_cuda = opt.cuda

# Map the words from one dictionary to another.
def sync_dicts(dict1, dict2):
    dict_map = torch.ones(len(dict1["index_to_symbol"])).long()
    for i in range(len(dict1["index_to_symbol"])):
        temp = dict1["index_to_symbol"][i]
        try:
            res = dict2["symbol_to_index"][temp]
            dict_map[i] = res or 0
        except Exception as e:
            dict_map[i] = 0

    return dict_map

def process_word(input_word):
    return re.sub(r'\d', '#', input_word.lower())

def find_k_max(k, mat):
    return torch.topk(mat, 2 * k)

def flat_to_rc(v, indices, flat_index):
    row = int(math.floor((flat_index) / v.size(1)))
    col = int((flat_index) % (v.size(1)))
    return row, indices[row][col]

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
    a_s2i = adict["symbol_to_index"]
    a_i2s = adict["index_to_symbol"]
    t_s2i = tdict["symbol_to_index"]
    t_i2s = tdict["index_to_symbol"]

    K = opt.beamSize
    FINAL_VAL = 1000
    INF = float('inf')

    UNK = t_s2i["<unk>"]
    START = t_s2i["<s>"]
    END = t_s2i["</s>"]

    W = mlp.window
    opt.window = mlp.window

    sent_num = 0
    for line in sent_file:
        print(line)

        sent_num += 1

        # Add padding
        true_line = "<s> <s> <s> {} </s> </s> </s>".format(line)
        words = true_line.split()

        article = torch.zeros(len(words))
        for j in range(0, len(words)):
            word = process_word(words[j])
            try:
                article[j] = a_s2i[word] or a_s2i["<unk>"]
                # print
            except Exception as e:
                article[j] = a_s2i["<unk>"]


        n = opt.length

        # Initilize the charts.
        # scores[i][k] is the log prob of the k'th hyp of i words.
        # hyps[i][k] contains the words in k'th hyp at
        #          i word (left padded with W <s>) tokens.
        result = []
        scores = torch.zeros(n+1, K).float()
        hyps = torch.zeros(n+1, K, W+n+1).long().fill_(START)

        # Initilialize used word set.
        # words_used[i][k] is a set of the words used in the i,k hyp.
        words_used = []

        if opt.blockRepeatWords:
            for i in range(n+1):
                words_used[i] = []
                for k in range(K):
                    words_used[i][k] = []

        for i in range(n):
            cur_beam = hyps[i]
            cur_beam = cur_beam.narrow(1, i+1, W)
            cur_K = K

            # (1) Score all next words for each context in the beam.
            #    log p(y_{i+1} | y_c, x) for all y_c
            input = data.make_input(article, cur_beam, cur_K)
            model_scores = mlp.forward(*input)
            out = model_scores.clone().double().mul(opt.lmWeight)

            # If length limit is reached, next word must be end.
            finalized = (i == n-1) and opt.fixedLength

            if finalized:
                out[:, END] += FINAL_VAL
            else:

                # Apply hard constraints
                out[:, START] = -INF
                if not opt.allowUNK:
                    out[:, UNK] = -INF

                if opt.fixedLength:
                    out[:, END] = -INF

                # TODO Add additional extractive features.
                # feat_gen.add_features(out, cur_beam)

            # Only take first row when starting out.
            if i == 0:
                cur_K = 1
                out.narrow(1, 1, 1)
                model_scores = model_scores.narrow(1, 1, 1)

            # Prob of summary is log p + log p(y_{i+1} | y_c, x)
            for k in range(cur_K):
                out[k] += scores[i][k]

            # (2) Retain the K-best words for each hypothesis using GPU.
            # This leaves a KxK matrix which we flatten to a K^2 vector.
            max_scores, mat_indices = find_k_max(K, apply_cuda(out))
            flat = max_scores.view(max_scores.size(0) * max_scores.size(1)).float()

            # 3) Construct the next hypotheses by taking the next k-best.
            seen_ngram = []
            for k in range(K):
                for _ in range(100):
                    # (3a) Pull the score, index, rank, and word of the
                    # current best in the table, and then zero it out.
                    score, index = torch.max(flat, 0)
                    score = float(score)
                    index = int(index)
                    if finalized:
                        score -= FINAL_VAL

                    scores[i+1][k] = score
                    prev_k, y_i1 = flat_to_rc(max_scores, mat_indices, index)
                    flat[index] = -INF

                    # (3b) Is this a valid next word?
                    blocked = opt.blockRepeatWords and words_used[i][prev_k][y_i1]

                    # HACK ignored stuff featuring feat_gen

                    # Hypothesis recombination
                    new_context = []
                    if opt.recombine:
                        for j in range(i+2, i+W):
                            new_context.append(hyps[i][prev_k][j])

                        new_context.append(y_i1)
                        # HACK should be deep find
                        blocked = blocked or (new_context in seen_ngram)

                    # (3c) Add the word, its score, and its features to the
                    # beam.
                    if not blocked:
                        # Update tables with new hypothesis
                        for j in range(i+W):
                            pword = hyps[i][prev_k][j]
                            hyps[i+1][k][j] = pword
                            if opt.blockRepeatWords:
                                words_used[i+1][k][pword] = True

                        hyps[i+1][k][i+W+1] = y_i1
                        if opt.blockRepeatWords:
                            words_used[i+1][k][y+i1] = True

                        # Keep track of hypotheses seen
                        if opt.recombine:
                            seen_ngram.append(new_context)

                        # Keep track of features used (For MERT)
                        # HACK NOPE

                        # If we have produced an END symbol, push to stack
                        if y_i1 == END:
                            result.append((i + 1, scores[i+1][k], hyps[i+1][k].clone()))
                            scores[i+1][k] = -INF




        sorted_results =  sorted(result, key=lambda a: a[1])


        if len(sorted_results) == 0:
            print("*FAIL*")
        else:
            # Print out in standard format
            length, _, output = sorted_results[1]
            final = "|->"
            for j in range(W+2, W+length-1):
                index = int(output[j])
                word = t_i2s[index]
                final += " {}".format(word)

            print(final)
main()
