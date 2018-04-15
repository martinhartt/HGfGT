import torch
import torch.nn as nn

import nnlm
import encoder
# import beam_search
import argparse
import util
import data
import re


parser = argparse.ArgumentParser(description='Train a summarization model.')

# beam.addOpts(parser)

parser.add_argument('-modelFilename',  default='', help='Model to test.')
parser.add_argument('-inputf',         default='', help='Input article files. ')
parser.add_argument('-nbest',type=bool,       default=False, help='Write out the nbest list in ZMert format.')
parser.add_argument('-length',type=int,          default=15, help='Maximum length of summary.')
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

    sent_num = 0
    for line in sent_file:
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

        print(article)

    print('OK')

    # Run beam search
    # sbeam = beam.Beam(opt, mlp.mlp, mlp.encoder_model, dict_map, tdict)
    # results = sbeam.generate(article, length)

main()
