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

# Make directory for output if it doesn't exist

try:
    os.mkdir(sys.argv[2] + "/raw/" + sys.argv[1].split("/")[-2])
except OSError:
    pass

# Strip off .gz ending
end = "/".join(sys.argv[1].split("/")[-2:])

out = open(sys.argv[2] + "/raw/" + end, "w")

# Parse and print titles and articles
NONE, HEAD, NEXT, TEXT = 0, 1, 2, 3
MODE = NONE


def normalize(sent):
    sent = sent.lower()
    sent = re.sub(r"([.!?])", r" \1", sent)
    sent = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", sent)
    sent = re.sub(r'\d', '#', sent)
    sent += " "
    return sent


title = ""
article = ""

raw = open(sys.argv[1]).read()

title_raw, article_raw = raw.split('\nTEXT:')[:2]

# Remove TITLE and TEXT labels and process
title = normalize(title_raw[6:].strip().replace('\n', ' '))
article = normalize(article_raw.strip().replace('\n', ' '))

# title \t article
out.write("{}\t{}\n".format(title, article))
