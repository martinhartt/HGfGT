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

import sys

for l in open(sys.argv[1]):
    splits = l.strip().split("\t")
    if len(splits) != 2:
        continue
    title, article = splits

    # No blanks.
    if title.strip() == "" or article.strip() == "":
        continue

    title_words = title.split()
    article_words = article.split()

    # Reasonable lengths
    if not (10 < len(article_words) < 100 and 3 < len(title_words) < 50):
        continue

    # Okay, print.
    print("{}\t{}".format(title, article))