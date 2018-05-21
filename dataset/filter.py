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

import argparse
import sys

parser = argparse.ArgumentParser(
    description='Evaluate the results of a model.')

parser.add_argument(
    'inputFile',
    default='',
    help='Input file')
parser.add_argument('--wordOverlap', type=bool, default=False, help='Word overlap')
parser.add_argument('--firstSent', type=bool, default=False, help='Only use first sentence?')
parser.add_argument('--lengthRange', type=bool, default=False, help='Length range?')

opt = parser.parse_args()

for l in open(opt.inputFile):
    splits = l.strip().split("\t")
    if len(splits) != 2:
        continue
    title, article = splits

    # Get only first sentence
    if opt.firstSent:
        try:
            sent_boundary = article.index('<sb> ')
            article = article[0:sent_boundary]
        except Exception as e:
            pass
    else:
        article = article.replace('<sb>', '')


    # No blanks.
    if title.strip() == "" or article.strip() == "":
        continue

    title_words = title.split()
    article_words = article.split()

    # Reasonable lengths
    if opt.lengthRange:
        if not (10 < len(article_words) < 100 and 3 < len(title_words) < 50):
            continue

    # Some word match.
    if opt.wordOverlap:
        matches = len(
            set([w.lower() for w in title_words if len(w) > 3]) & set(
                [w.lower() for w in article_words if len(w) > 3]))
        if matches < 1:
            continue

    # Okay, print.
    print("{}\t{}".format(title, article))
