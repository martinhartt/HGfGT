import torch
import argparse

parser = argparse.ArgumentParser(
    description='Build torch serialized version of a summarization problem.')

parser.add_argument('-inputFile', default='', help='The input file.')
parser.add_argument('-inDictionary', default='', help='The input dictionary.')
parser.add_argument('-outDirectory', default='', help='The output directory.')
parser.add_argument('-outPrefix', default='', help='The output prefix.')

opt = parser.parse_args()


def encode(sentence, w2i):
    return torch.tensor([w2i.get(word, w2i['<unk>']) for word in sentence.split()])

def main():
    dict = torch.load(opt.inDictionary)

    titleTensors = []
    articleTensors = []

    for l in open(opt.inputFile):
        title, article = l.strip().split('\t')

        titleTensors.append(encode(title, dict["w2i"]))
        articleTensors.append(encode(article, dict["w2i"]))

    print("Saving inputs...")
    torch.save(titleTensors, '{}/{}.article.torch'.format(
        opt.outDirectory, opt.outPrefix))
    torch.save(articleTensors, '{}/{}.title.torch'.format(
        opt.outDirectory, opt.outPrefix))


main()
