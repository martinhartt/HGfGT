import torch


def apply_cuda(obj):
    return obj.cuda() if torch.cuda.is_available() else obj


def encode(sentence, w2i):
    return torch.tensor([w2i.get(word, w2i['<unk>']) for word in sentence.split()])

def decode(tensor, i2w):
    return " ".join([i2w.get(int(t), "<unk>") for t in tensor])
