from nnlm import train
from data import load_title, load_article, Data
import encoder

import argparse

parser = argparse.ArgumentParser(description='Train a summarization model.')

data.add_opts(parser)
encoder.add_opts(parser)
nnlm.addOpts(parser)

opt = parser.parse_args()

def main():
    tdata = load_title(opt.titleDir, True)
    article_data = load_article(opt.articleDir)

    valid_data = load_title(opt.validTitleDir, None, tdata["dict"])
    valid_article_data = load_article(opt.validArticleDir, article_data["dict"])

    # Make main LM
    train_data = Data(tdata, article_data)
    valid = Data(valid_data, valid_article_data)

    encoder_mlp = encoder.AttnBowEncoder(opt.bowDim, opt.window, len(train_data.title_data["dict"]["index_to_symbol"]), len(train_data.article_data["dict"]["index_to_symbol"]), opt)

    mlp = nnlm.NNLM(opt, tdata["dict"], encoder_mlp, opt.bowDim, article_data["dict"])
    mlp.train(train_data, valid)

main()
