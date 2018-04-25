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

#/usr/bin/env python

import sys
import os
import re
import gzip
#@lint-avoid-python-3-compatibility-imports

# Make directory for output if it doesn't exist

try:
    os.mkdir(sys.argv[2] + "/" + sys.argv[1].split("/")[-2])
except OSError:
    pass

# Strip off .gz ending
end = "/".join(sys.argv[1].split("/")[-2:])[:-len(".gz")] + ".txt"

out = open(sys.argv[2] + "/" + end, "w")

# Parse and print titles and articles
NONE, HEAD, NEXT, TEXT = 0, 1, 2, 3
MODE = NONE
title_parse = ""
article_parse = []

# FIX: Some parses are mis-parenthesized.
def fix_paren(parse):
    if len(parse) < 2:
        return parse
    if parse[0] == "(" and parse[1] == " ":
        return parse[2:-1]
    return parse


def normalize(sent):
    print('BEFORE', sent)
    sent = re.sub(r"([.!?])", r" \1", sent)
    sent = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", sent)
    sent = re.sub(r'\d', '#', sent)
    print('AFTER', sent)
    return sent

for l in gzip.open(sys.argv[1]):
    if MODE == HEAD:
        title_parse = normalize(fix_paren(l.strip()))
        MODE = NEXT

    if MODE == TEXT and l.strip() != "</P>":
        article_parse.append(normalize(fix_paren(l.strip())))

    if MODE == NONE and l.strip() == "<HEADLINE>":
        MODE = HEAD

    if MODE == NEXT and l.strip() == "<P>":
        MODE = TEXT

    if MODE == TEXT and l.strip() == "</P>":
        articles = []
        # Annotated gigaword has a poor sentence segmenter.
        # Ensure there is a least a period.

        for i in range(len(article_parse)):
            articles.append(article_parse[i])
            if "(. .)" in article_parse[i]:
                break

        article_parse = "(TOP " + " ".join(articles) + ")"

        # title_parse \t article_parse \t title \t article
        print >>out, "\t".join([title_parse, article_parse,
                                title_parse,
                                " ".join(articles)])
        article_parse = []
        MODE = NONE
