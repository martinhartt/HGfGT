from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.tokenizers import Tokenizer
import sys

def leadSummariser(document, no_of_sents):
    for sent in document.sentences[:no_of_sents]:
        yield str(sent)

summarisers = {
    "lead": leadSummariser,
    "luhn": LuhnSummarizer(),
    "lsa": LsaSummarizer(),
    "lex_rank": LexRankSummarizer(),
    "text_rank": TextRankSummarizer(),
    "sum_basic": SumBasicSummarizer(),
    "kl": KLSummarizer()
}

tokenizer = Tokenizer("english")

def extractive(article, title=None):
    raw = article.replace(' <sb>', '').strip()

    parser = PlaintextParser.from_string(raw, tokenizer)

    summs = []
    for name, summariser in summarisers.items():
        temp = ""
        for sentence in summariser(parser.document, 4):
            temp += " {}".format(sentence)
            if len(tokenizer.to_words(temp)) > 50:
                break

        summs.append(temp)

    result = "\t".join(summs)

    if title is not None:
        return "{}\t{}".format(title, result.strip())
    else:
        return result.strip()


if __name__ == '__main__':
    line = sys.argv[1]
    if line.strip() == "":
        print("")
        exit(0)

    if len(line.split('\t')) > 2:
        print("")
        exit(0)

    components = line.split('\t')
    if len(components) == 2:
        title, article = components
    elif len(components) == 1:
        article = components[0]
        title = None
    else:
        exit()

    out = open(sys.argv[2], "a")
    out.write("{}\n".format(extractive(article, title)))
    out.close()
