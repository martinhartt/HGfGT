import torch
import argparse
from util import apply_cuda
import data
import re
import math

from extractive import extractive

from data import AbsDataLoader, HeirDataLoader

parser = argparse.ArgumentParser(description='Train a summarization model.')

parser.add_argument('--model', default='', help='Model to test.')
parser.add_argument('--inputf', default='', help='Input article files. ')
parser.add_argument('--outputf', default='', help='Actual title files. ')
parser.add_argument(
    '--length', type=int, default=15, help='Maximum length of summary.')
parser.add_argument(
    '--allowUNK', type=bool, default=False, help="Allow generating <unk>.")
parser.add_argument(
    '--fixedLength',
    type=bool,
    default=True,
    help="Produce exact no of words.")
parser.add_argument(
    '--blockRepeatWords',
    type=bool,
    default=False,
    help="Disallow generating a word twice.")
parser.add_argument(
    '--lmWeight', type=float, default=1.0, help="Weight for main model.")
parser.add_argument(
    '--beamSize', type=int, default=100, help="Size of the beam.")
parser.add_argument('--dictionary', default='', help='The input dictionary.')

data.add_opts(parser)

opt = parser.parse_args()

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

def decode(sent, i2w):
    return [i2w.get(ix, "<unk>") for ix in sent.split()]


def set_hard_constraints(out_scores, w2i, finalized):
    INF = float('inf')

    # Apply hard constraints
    out_scores[:, w2i["<s>"]] = -INF
    out_scores[:, w2i["<null>"]] = -INF
    out_scores[:, w2i["<sb>"]] = -INF

    if not opt.allowUNK:
        out_scores[:, w2i["<unk>"]] = -INF

    if opt.fixedLength:
        out_scores[:, w2i["</s>"]] = -INF

def main():
    state = torch.load(opt.model)

    if opt.heir:
        mlp, encoder = state
    else:
        mlp = state

    dict = data.loadDict(opt.dictionary)

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

    actual = open(opt.outputf).read().split('\n')

    sent_num = 0
    for line in sent_file:
        if line.strip() == "":
            continue

        # Add padding
        if opt.heir:
            summaries = extractive(line).split("\t")
            print("\n> {}...".format(summaries[0]))
            encoded_summaries = [encode(normalize(summary), w2i) for summary in summaries]
            article = HeirDataLoader.torchify(encoded_summaries, variable=True, revsort=True)
        else:
            print("\n> {}".format(line))
            true_line = "<s> <s> <s> {} </s> </s> </s>".format(normalize(line))

            article = torch.tensor(encode(true_line, w2i))

        n = opt.length

        hyps = apply_cuda(torch.zeros(K, W + n).long().fill_(w2i["<s>"]))
        scores = apply_cuda(torch.zeros(K).float())

        for step in range(n):
            new_candidates = []

            start = 0 if opt.heir else step
            end = step+W
            context = hyps[:, start:end] # context

            if opt.heir:
                encoder_out = encoder(article)
                model_scores = mlp(encoder_out, (context, [step+1] * K))
            else:
                article_t, context_t = AbsDataLoader.make_input(article, context, K)
                model_scores = mlp(article_t, context_t)

            out_scores = model_scores.data

            # Apply hard constraints
            finalized = (step == n - 1) and opt.fixedLength
            # set_hard_constraints(out_scores, w2i, finalized)

            for sample in range(K): # Per certain context
                if context[sample][-1] == w2i["</s>"]:
                    end = w2i["</s>"]
                    combined = torch.cat((context[sample], apply_cuda(torch.tensor([end]))))
                    candidate = [combined, scores[sample]]
                    new_candidates.append(candidate)
                    continue

                top_scores, top_indexes = torch.topk(out_scores[sample], K)

                for ix, score in zip(top_indexes, top_scores):
                    combined = torch.cat((context[sample], apply_cuda(torch.tensor([ix]))))
                    candidate = [combined, scores[sample] + score]
                    new_candidates.append(candidate)

            ordered = list(reversed(sorted(new_candidates, key=lambda cand:cand[1])))
            h, s = zip(*ordered)

            for r in range(K):
                print(end+1-start)
                print(h[r])
                hyps[r][start:end+1] = h[r]
                scores[r] = s[r]

        s, top_ixs = torch.topk(scores, 1)

        final = hyps[int(top_ixs)][W:-1]

        print("= {}".format(actual[sent_num]))
        print("< {}".format(" ".join([i2w[int(ix)] for ix in final])))
        print("")

        sent_num += 1

main()
