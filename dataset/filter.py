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
import spacy

for l in open(sys.argv[1]):
    splits = l.strip().split("\t")
    if len(splits) != 2:
        continue
    title, article = splits

    # Get only first sentence
    try:
        sent_boundary = article.index('. ')
        article = article[0:sent_boundary]
    except Exception as e:
        pass

    # No blanks.
    if title.strip() == "" or article.strip() == "":
        continue

    title_words = title.split()
    article_words = article.split()

    # Reasonable lengths
    if not (10 < len(article_words) < 100 and 3 < len(title_words) < 50):
        continue

    # Some word match.
    matches = len(
        set([w.lower() for w in title_words if len(w) > 3]) & set(
            [w.lower() for w in article_words if len(w) > 3]))
    if matches < 1:
        continue

    # Okay, print.
    print("{}\t{}".format(title, article))
