import torch
import os
cur_dir = os.path.dirname(__file__)

def build_glove(w2i, size=300):
    weights = torch.randn(len(w2i) + 1, size)

    print("Processing GloVe word2vec")
    for line in open(os.path.join(cur_dir, '..', 'data/glove.42B.{}d.txt'.format(size))):
        components = line.split()
        word = components[0]
        if word in w2i:
            vec = [float(c) for c in components[1:]]
            print(word, vec)
            weights[w2i[word]] = torch.tensor(vec)

    return weights
