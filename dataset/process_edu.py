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

end = "/".join(sys.argv[1].split("/")[-2:])


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

def get_words(parse):
    return parse.split(' ')

def remove_digits(parse):
    return re.sub(r'\d', '#', parse)

raw = open(sys.argv[1]).read()

title_raw, article_raw = raw.split('\n\n')[:2]

# Remove TITLE and TEXT labels and process
title = title_raw[6:].strip().replace('\n', ' ')
article = article_raw[5:].strip().replace('\n', ' ').split('.')[0]

# title_parse \t article_parse \t title \t article
print >>out, "\t".join([title, "(TOP {})".format(article),
                        title,
                        article])
