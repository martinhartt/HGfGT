import nnlm
import data
import encoder

import argparse

parser = argparse.ArgumentParser(description='Train a summarization model.')

data.add_opts(parser)
encoder.add_opts(parser)
nnlm.addOpts(parser)

opt = parser.parse_args()


def main():
    print("Loading dictionary...")
    dict = data.load(opt.workingDir, train=True, type='dict', heir=opt.heir, small=opt.small)

    print("Loading training data...")
    train_title = data.load(
        opt.workingDir, train=True, type="title", heir=opt.heir, small=opt.small)


    train_article = data.load(
        opt.workingDir, train=True, type="article", heir=opt.heir, small=opt.small)

    print("Loading validation data...")
    valid_title = data.load(
        opt.workingDir, train=False, type="title", heir=opt.heir, small=opt.small)
    valid_article = data.load(
        opt.workingDir, train=False, type="article", heir=opt.heir, small=opt.small)

    # Make main LM
    print("Constructing train tensors...")
    DataLoader = data.HeirDataLoader if opt.heir else data.AbsDataLoader

    train_data = DataLoader(train_title, train_article, dict, window=opt.window)

    print("Constructing validation tensors...")
    valid_data = DataLoader(valid_title, valid_article, dict, window=opt.window)

    print("Setting up language model and training parameters...")
    mlp = nnlm.NNLM(opt, dict)

    print("Training...")
    mlp.train(train_data, valid_data)


main()
