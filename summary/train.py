import nnlm
import data
import encoder

import argparse

parser = argparse.ArgumentParser(description='Train a summarization model.')

parser.add_argument('-trainFile', default='', help='The input training file.')
parser.add_argument('-validFile', default='', help='The input validation file.')
parser.add_argument('-dictionary', default='', help='The input dictionary.')

data.add_opts(parser)
encoder.add_opts(parser)
nnlm.addOpts(parser)

opt = parser.parse_args()


def main():
    print("Loading dictionary...")
    print(opt.dictionary)
    dict = data.loadDict(opt.dictionary)

    DataLoader = data.HeirDataLoader if opt.heir else data.AbsDataLoader

    print("Constructing train tensors...")
    train_data = DataLoader(opt.trainFile, dict, window=opt.window, maxSize=opt.maxSize)

    print("Constructing validation tensors...")
    valid_data = DataLoader(opt.validFile, dict, window=opt.window, maxSize=opt.maxSize)

    print("Setting up language model and training parameters...")
    mlp = nnlm.NNLM(opt, dict)

    print("Training...")
    mlp.train(train_data, valid_data)


main()
