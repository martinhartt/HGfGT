import torch
import sys

inputDict = sys.argv[1]
outputFile = sys.argv[2]

word_id = 0
dict = {"w2i": {}, "i2w": {}}

for l in open(inputDict):
    word = l.split()[0]
    dict["w2i"][word] = word_id
    dict["i2w"][word_id] = word

    word_id += 1

torch.save(dict, outputFile)
