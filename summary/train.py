import trainer
import data

import argparse

parser = argparse.ArgumentParser(description='Train a summarization model.')

parser.add_argument('--train', default='', help='The input training file.')
parser.add_argument('--valid', default='', help='The input validation file.')
parser.add_argument('--dictionary', default='', help='The input dictionary.')
parser.add_argument('--pooling', default=False, help='Enable pooling?')

parser.add_argument('--bowDim', type=int, default=300, help="Article embedding size.")
parser.add_argument(
    '--attenPool',
    type=int,
    default=5,
    help="Attention model pooling size.")
parser.add_argument(
    '--glove',
    type=bool,
    default=False,
    help="Use pretrained GloVe embeddings"
)

data.add_opts(parser)
trainer.addOpts(parser)

opt = parser.parse_args()


def main():
    print("Loading dictionary...")
    print(opt.dictionary)
    dict = data.loadDict(opt.dictionary)

    DataLoader = data.HeirDataLoader if opt.heir else data.AbsDataLoader

    print("Constructing train tensors...")
    train_data = DataLoader(opt.train, dict, window=opt.window, maxSize=opt.maxSize)

    print("Constructing validation tensors...")
    valid_data = DataLoader(opt.valid, dict, window=opt.window, maxSize=opt.maxSize)

    print("Setting up language model and training parameters...")
    t = trainer.Trainer(opt, dict)

    print("Training...")
    t.train(train_data, valid_data)


main()
