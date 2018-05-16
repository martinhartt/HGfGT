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

    titleList = []
    articleList = []

    for l in open(opt.inputFile):
        components = l.strip().split('\t')

        title = components[0]
        articles = components[1:]

        titleList.append(encode(title, dict["w2i"]))

        if len(articles) > 1:
            articleList.append([encode(article, dict["w2i"]) for article in articles])
        elif len(articles) > 0:
            articleList.append(encode(articles[0], dict["w2i"]))

    print("Saving inputs...")
    torch.save(articleList, '{}/{}.article.torch'.format(
        opt.outDirectory, opt.outPrefix))
    torch.save(titleList, '{}/{}.title.torch'.format(
        opt.outDirectory, opt.outPrefix))


main()
