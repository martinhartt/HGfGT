#  Copyright (c) 2018, Martin Hartt
#
#  Copyright (c) 2015, Facebook, Inc.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree. An additional grant
#  of patent rights can be found in the PATENTS file in the same directory.
#
#  Author: Alexander M Rush <srush@seas.harvard.edu>
#          Sumit Chopra <spchopra@fb.com>
#          Jason Weston <jase@fb.com>


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
parser.add_argument(
    '--extraAttnLinear',
    type=bool,
    default=False,
    help="Add extra linear layer for attn?"
)
parser.add_argument(
    '--noAttn',
    type=bool,
    default=False,
    help="Disable attn for hier"
)
parser.add_argument(
    '--simple',
    type=bool,
    default=False,
    help="Enable simple mode"
)

data.add_opts(parser)
trainer.add_opts(parser)

opt = parser.parse_args()


def main():
    print("Loading dictionary...")
    print(opt.dictionary)
    dict = data.load_dict(opt.dictionary)

    DataLoader = data.HierDataLoader if opt.hier else data.AbsDataLoader

    print("Constructing train tensors...")
    train_data = DataLoader(opt.train, dict, opt, window=opt.window, max_size=opt.maxSize)

    print("Constructing validation tensors...")
    valid_data = DataLoader(opt.valid, dict, opt, window=opt.window, max_size=opt.maxSize)

    print("Setting up language model and training parameters...")
    t = trainer.Trainer(opt, dict)

    print("Training...")
    t.train(train_data, valid_data)


main()
