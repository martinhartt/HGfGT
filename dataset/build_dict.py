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
import sys

inputDict = sys.argv[1]
outputFile = sys.argv[2]

word_id = 0
dict = {"w2i": {}, "i2w": {}}

for l in open(inputDict):
    word = l.split()[0]
    dict["w2i"][word] = word_id
    dict["i2w"][word_id] = word

    word_id += 1

torch.save(dict, outputFile)
