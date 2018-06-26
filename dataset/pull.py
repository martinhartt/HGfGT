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
"""
Pull out elements of the title-article file.
"""
import sys
import re

INPUT_FILENAME = sys.argv[1]
no_dict = len(sys.argv) < 3

if not no_dict:
    DICT_FILENAME = sys.argv[2]
    dict = set([l.split()[0] for l in open(DICT_FILENAME)])

article_out = open(re.sub(r"data", r"article", INPUT_FILENAME), "w")
title_out = open(re.sub(r"data", r"title", INPUT_FILENAME), "w")

for l in open(INPUT_FILENAME):
    splits = l.strip().split("\t")

    title = splits[0]
    article_components = splits[1:]

    article_result = ""
    for article in article_components:
        article_words = [w if no_dict or w in dict else "<unk>" for w in article.split()]
        article_result += " ".join(article_words)
    article_out.write("{}\n".format(article_result))

    title_words = [w if no_dict or w in dict else "<unk>" for w in title.split()]
    title_out.write(" ".join(title_words))
    title_out.write("\n")
