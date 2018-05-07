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
    dict = data.load(opt.workingDir, 'dict', opt.filter)

    train_title = data.load(opt.workingDir, train=True, type="title", filter=opt.filter)
    train_article = data.load(opt.workingDir, train=True, type="article", filter=opt.filter)

    valid_title = data.load(opt.workingDir, train=False, type="title", filter=opt.filter)
    valid_article = data.load(opt.workingDir, train=False, type="article", filter=opt.filter)

    # Make main LM
    train_data = data.Data(train_title, train_article, dict, opt.batchSize)
    valid_data = data.Data(valid_title, valid_article, dict, opt.batchSize)

    encoder = encoder.AttnBowEncoder(opt.bowDim, opt.window, len(dict["index_to_symbol"])))

    mlp = nnlm.NNLM(opt, dict, encoder, opt.bowDim)
    mlp.train(train_data, valid_data)

main()
