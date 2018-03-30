

def addOpts(parser):
    parser.add_argument('-lmWeight',      default=1.0, help="Feature weight for the neural model.")
    parser.add_argument('-unigramBonus',  default=0.0, help="Feature weight for unigram extraction.")
    parser.add_argument('-bigramBonus',   default=0.0, help="Feature weight for bigram extraction.")
    parser.add_argument('-trigramBonus',  default=0.0, help="Feature weight for trigram extraction.")
    parser.add_argument('-lengthBonus',   default=0.0, help="Feature weight for length.")
    parser.add_argument('-unorderBonus',  default=0.0, help="Feature weight for out-of-order.")

-- Feature positions.
NNLM = 1
UNI  = 2
BI   = 3
TRI  = 4
OO   = 5
LEN  = 6

kFeat = 6

class Features(object):
    """docstring for features."""
    def __init__(self, opt, article_to_title):
        super(features, self).__init__()

        self.opt = opt
        self.num_features = kFeat
        self.article_to_title = article_to_title

    def match_words(self, START, article):
        self.ooordered_ngram = []
        ordered_ngram = []
        self.ngrams = [[], [], []]
        hist = [START, START, START, START]

        for j in range(0, article.size(0)):
            tw = self.article_to_title[article[j]]

            # Does the current word exist in title dict
            # TODO Figure out whats going on here
            if tw != None:
                for j2 in range(0, j):
                    tw2 = self.article_to_title[article[j2]]
                    if tw2 != None:
                        util.add(ordered_ngram, [tw2, tw])
                        if not util.has(ordered_ngram, [tw, tw2]):
                            util.add(self.ooordered_ngram, [tw, tw2])

            # Advance window
            for k = range(1, 3):
                hist[k-1] = hist[k]

            hist[3] = tw
