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
from collections import Counter

words = Counter()
limit = int(sys.argv[3])

for l in open(sys.argv[1]):
    splits = l.strip().split("\t")
    if len(splits) != 2:
        continue

    title, article = l.strip().split("\t")

    words.update(title.lower().split())
    words.update(article.lower().split())

with open(sys.argv[2] + ".dict", "w") as f:
    f.write("<unk> {}\n".format(1e5))
    f.write("<s> {}\n".format(1e5))
    f.write("</s> {}\n".format(1e5))
    f.write("<sb> {}\n".format(1e5))

    for word, count in words.most_common():
        if count < limit:
            break
        f.write("{} {}\n".format(word, count))
