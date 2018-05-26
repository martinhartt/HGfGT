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

    result = ""
    for name, summariser in summarisers.items():
        for sentence in summariser(parser.document, 4):
            result += " {}".format(sentence)
            if len(tokenizer.to_words(result)) > 50:
                break

        result += "\t"

    if title is not None:
        return "{}\t{}".format(title, result.strip())
    else:
        return result.strip()


if __name__ == '__main__':
    line = sys.argv[1]
    if line.strip() == "":
        print("")

    if len(line.split('\t')) > 2:
        print("")

    title, article = line.split('\t')

    out = open(sys.argv[2], "a")
    out.write("{}\n".format(extractive(article, title)))
    out.close()
