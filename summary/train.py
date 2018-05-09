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
    dict = data.load(opt.workingDir, train=True, type='dict', filter=opt.filter)

    print("Loading training data...")
    train_title = data.load(
        opt.workingDir, train=True, type="title", filter=opt.filter)
    train_article = data.load(
        opt.workingDir, train=True, type="article", filter=opt.filter)

    print("Loading validation data...")
    valid_title = data.load(
        opt.workingDir, train=False, type="title", filter=opt.filter)
    valid_article = data.load(
        opt.workingDir, train=False, type="article", filter=opt.filter)

    # Make main LM
    print("Constructing train tensors...")
    train_data = data.Data(train_title, train_article, dict, opt.window)

    print("Constructing validation tensors...")
    valid_data = data.Data(valid_title, valid_article, dict, opt.window)


    print("Building encoder...")
    attn_encoder = encoder.AttnBowEncoder(opt.bowDim, opt.window, len(dict["i2w"]), opt)

    print("Setting up language model and training parameters...")
    mlp = nnlm.NNLM(opt, dict, attn_encoder, opt.bowDim)

    print("Training...")
    mlp.train(train_data, valid_data)


main()
