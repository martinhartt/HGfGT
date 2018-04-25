import torch
import sys

inputDict = sys.argv[1]
outputFile = sys.argv[2]

word_id = 0
dict = {
    "symbol_to_index": {},
    "index_to_symbol": {}
}

for l in open(inputDict):
    word = l.split()[0]
    dict["symbol_to_index"][word] = word_id
    dict["index_to_symbol"][word_id] = word

    word_id += 1

torch.save(dict, outputFile)
