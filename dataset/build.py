import torch
import argparse
from collections import Counter

parser = argparse.ArgumentParser(description='Build torch serialized version of a summarization problem.')

parser.add_argument('-window', default=5, type=int, help='The ngram window to use.')

parser.add_argument('-inTitleFile', default='',       help='The input file.')
parser.add_argument('-inTitleDictionary', default='', help='The input dictionary.')
parser.add_argument('-outTitleDirectory', default='', help='The output directory.')
parser.add_argument('-inArticleFile', default='',     help='The input file.')
parser.add_argument('-inArticleDictionary', default='', help='The input dictionary.')
parser.add_argument('-outArticleDirectory', default='', help='The output directory.')

opt = parser.parse_args()

def count(file, aligned_lengths, pad):
    # Count up properties of the input file.
    f = open(file, 'r')

    counter = {
        "nsents": 0,
        "aligned_lengths": Counter(),
        "line_lengths": Counter(),
        "bucket_words": Counter()
    }
    nline = 0

    for l in f:
        true_l = l
        if pad:
            true_l = "<s> <s> <s> {} </s> </s> </s>".format(l)

        line = true_l.split()

        counter["line_lengths"][len(line)] += 1
        counter["nsents"] += 1
        counter["aligned_lengths"][nline] += len(line)

        if aligned_lengths != None:
            # Add extra for implicit </s>
            counter["bucket_words"][aligned_lengths[nline]] += len(line) + 1

        nline += 1

    return counter

def build_article_matrices(dict, file, nsents, line_lengths):
    # For each length bucket, construct a #sentence x length matrix
    # of word forms.
    f = open(file, 'r')

    # One matrix for each length
    mat = {}

    # Number of sentences seen of this length
    of_length = {}

    for length, count in line_lengths.iteritems():
        mat[length] = torch.LongTensor(count, length).zero_()
        of_length[length] = 0

    # For each sentence.
    # Col 1 is its length bin.
    # Col 2 is its position in bin.
    pos = torch.LongTensor(nsents, 2).zero_()

    nsent = 0
    for l in f:
        true_l = "<s> <s> <s> {} </s> </s> </s>".format(l)
        line = true_l.split()
        length = len(line)
        nbin = of_length[length]

        # Loop through words
        for j in range(0, len(line)):
            index = dict["symbol_to_index"].get(line[j], 0)
            mat[length][nbin][j] = index

        pos[nsent][0] = length
        pos[nsent][1] = nbin
        of_length[length] = nbin + 1
        nsent += 1

    return mat, pos

def build_title_matrices(dict, file, aligned_lengths, bucket_sizes, window):
    # For each article length bucket construct a num-words x 1 flat vector
    # of word forms and a corresponding num-words x window matrix of
    # context forms.
    nsent = 0
    pos = {}

    # One matrix for each length
    mat = {}
    ngram = {}

    # Number of sentences seen of this length.
    sent_of_length = {}
    words_of_length = {}

    # Initialize
    for length, count in bucket_sizes.iteritems():
        mat[length] = torch.LongTensor(count, 3).zero_()
        sent_of_length[length] = 0
        words_of_length[length] = 0
        ngram[length] = torch.LongTensor(count, window).zero_()

    # Columns are the preceding window.
    nline = 0
    f = open(file, 'r')
    for l in f:
        # Add implicit </s>
        true_l = "{} </s>".format(l)
        line = true_l.split()

        last = []
        # Initialize window as START symbol.
        for w in range(0, window):
            last.append(dict["symbol_to_index"]["<s>"])

        aligned_length = aligned_lengths[nline]

        for j in range(0, len(line)):
            nwords = words_of_length[aligned_length]

            index = dict["symbol_to_index"].get(line[j], 0)

            mat[aligned_length][nwords][0] = index
            mat[aligned_length][nwords][1] = sent_of_length[aligned_length]
            mat[aligned_length][nwords][2] = j

            # Move the window forward
            for w in range(0, window - 1):
                ngram[aligned_length][nwords][w] = last[w]
                last[w] = last[w+1]

            ngram[aligned_length][nwords][window-1] = last[window-1]
            last[window-1] = index
            words_of_length[aligned_length] = words_of_length[aligned_length] + 1

        sent_of_length[aligned_length] += 1
        nsent += 1

        nline += 1

    return mat, pos, ngram


def main():
    counter = count(opt.inArticleFile, None, True)
    dict = torch.load(opt.inArticleDictionary)

    # Contruct a rectangular word matrix.
    word_mat, offset_mat = build_article_matrices(
        dict,
        opt.inArticleFile,
        counter["nsents"],
        counter["line_lengths"]
    )

    torch.save(word_mat, '{}/word.mat.torch'.format(opt.outArticleDirectory))
    torch.save(offset_mat, '{}/offset.mat.torch'.format(opt.outArticleDirectory))

    title_counter = count(opt.inTitleFile, counter["aligned_lengths"], False)
    title_dict = torch.load(opt.inTitleDictionary)

    # Construct a 1d word matrix
    word_mat, offset_mat, ngram_mat = build_title_matrices(
        title_dict,
        opt.inTitleFile,
        counter["aligned_lengths"],
        title_counter["bucket_words"],
        opt.window
    )

    torch.save(word_mat, '{}/word.mat.torch'.format(opt.outTitleDirectory))
    torch.save(offset_mat, '{}/offset.mat.torch'.format(opt.outTitleDirectory))
    torch.save(ngram_mat, '{}/ngram.mat.torch'.format(opt.outTitleDirectory))

main()
