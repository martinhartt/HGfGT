import nnlm
import data
import encoder

import argparse

parser = argparse.ArgumentParser(description='Train a summarization model.')

data.add_opts(parser)
encoder.add_opts(parser)
nnlm.addOpts(parser)

opt = parser.parse_args()
data.enable_cuda = opt.cuda

def main():
    tdata = data.load_title(opt.titleDir, True)
    article_data = data.load_article(opt.articleDir)

    valid_data = data.load_title(opt.validTitleDir, None, tdata["dict"])
    valid_article_data = data.load_article(opt.validArticleDir, article_data["dict"])

    # Make main LM
    train_data = data.init(tdata, article_data)
    valid = data.init(valid_data, valid_article_data)

    encoder_mlp = encoder.AttnBowEncoder(opt.bowDim, opt.window, len(train_data.title_data["dict"]["index_to_symbol"]), len(train_data.article_data["dict"]["index_to_symbol"]), opt)

    mlp = nnlm.NNLM(opt, tdata["dict"], encoder_mlp, opt.bowDim, article_data["dict"])

    mlp.train(train_data, valid)

main()
