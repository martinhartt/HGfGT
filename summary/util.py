import random
import torch

def string_split(s, c = ' '):
    return s.split(c)

def shuffleTable(arr):
    random.shuffle(arr)

def apply_cuda(obj):
    return obj.cuda() if torch.cuda.is_available() else obj

def add(tab, keys):
    cur = tab

    for i in range(0, len(key) - 1):
        new_cur = cur[key[i]]

        if new_cur == None:
            cur[key[i]] = []

            new_cur = cur[key[i]]

        cur = new_cur

    cur[key[-1]] = True

def has(tab, keys):
    cur = tab
    for i in range(0, len(key)):
        cur = cur[key[i]]
        if cur == None:
            return False

    return True
