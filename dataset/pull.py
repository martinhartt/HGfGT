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
ARTICLE_DICT_FILENAME = sys.argv[2]
TITLE_DICT_FILENAME = sys.argv[3]

article_dict = set([l.split()[0] for l in open(ARTICLE_DICT_FILENAME)])
title_dict = set([l.split()[0] for l in open(TITLE_DICT_FILENAME)])

article_out = open(re.sub(r"data", r"article", INPUT_FILENAME), "w")
title_out = open(re.sub(r"data", r"title", INPUT_FILENAME), "w")

for l in open(INPUT_FILENAME):
    splits = l.strip().split("\t")

    if len(splits) != 2:
        continue

    title, article = splits

    article_words = [
        w if w in article_dict else "<unk>" for w in article.split()
    ]
    article_out.write(" ".join(article_words))
    article_out.write("\n")

    title_words = [w if w in title_dict else "<unk>" for w in title.split()]
    title_out.write(" ".join(title_words))
    title_out.write("\n")
