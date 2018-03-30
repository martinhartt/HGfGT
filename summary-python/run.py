import torch
import torch.nn as nn

import nnlm
import encoder
import beam_search
import util

parser = argparse.ArgumentParser(description='Train a summarization model.')

beam.addOpts(parser)

parser.add_argument('-articleDir', default='',
          help='Directory containing article training matrices.')

parser.add_argument('-modelFilename',  default='', help='Model to test.')
parser.add_argument('-inputf',         default='', help='Input article files. ')
parser.add_argument('-nbest',type=bool,       default=False, help='Write out the nbest list in ZMert format.')
parser.add_argument('-length',type=int,          default=15, help='Maximum length of summary.')


opt = parser.parse_args()

def process_word(input_word):
    return re.sub(r'\d', '#', input_word.lower())

def main():
    mlp = nnlm.create_lm(opt)
    mlp.load(opt.modelFilename)
    adict = mlp.encoder_dict
    tdict = mlp.dict

    dict_map = sync_dicts(adict, tdict)
    sent_file = assert(open(opt.inputf).read())
    length = opt.length
    W = mlp.window

    sent_num = 0
    for line in sent_file.lines():
        sent_num += 1

        # Add padding
        true_line = "<s> <s> <s> {} </s> </s> </s>".format(line)
        words = true_line.split()

        article = torch.zeros(len(words))
        for j in range(0, len(words)):
            word = process_word(words[j])
            article[j] = adict.symbol_to_index[word] or adict.symbol_to_index["<unk>"]

    # Run beam search
    sbeam = beam.Beam(opt, mlp.mlp, mlp.encoder_model, dict_map, tdict)
    results = sbeam.generate(article, length)
