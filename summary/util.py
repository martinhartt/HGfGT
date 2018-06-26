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


import torch

def apply_cuda(obj):
    return obj.cuda() if torch.cuda.is_available() else obj


def encode(sentence, w2i):
    return torch.tensor([w2i.get(word, w2i['<unk>']) for word in sentence.split()])

def decode(tensor, i2w):
    return " ".join([i2w.get(int(t), "<unk>") for t in tensor])
