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


# Make directory for output if it doesn't exist

try:
    os.mkdir(sys.argv[2] + "/raw/" + sys.argv[1].split("/")[-2])
except OSError:
    pass

# Strip off .gz ending
end = "/".join(sys.argv[1].split("/")[-2:])[:-len(".gz")] + ".txt"

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

for l in gzip.open(sys.argv[1]):
    line = l.strip()
    if MODE == HEAD:
        title += normalize(line)
        MODE = NEXT

    if MODE == TEXT and line == "</P>":
        article += "<sb> "

    if MODE == TEXT and len(line) > 0 and line[0] != "<":
        article += normalize(line)

    if MODE == NONE and line == "<HEADLINE>":
        MODE = HEAD

    if MODE == NEXT and line == "<P>":
        MODE = TEXT

    if MODE == TEXT and line == "</TEXT>":
        title = re.sub(r'\s\s', ' ', title).strip()
        article = re.sub(r'\s\s', ' ', article).strip()

        out.write("{}\t{}\n\n".format(title, article))

        title = ""
        article = ""

        MODE = NONE
