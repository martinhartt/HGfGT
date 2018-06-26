#  Copyright (c) 2018, Martin Hartt

import sys
import os

dataset = sys.argv[1]

proportions = {"train": 0.9, "test": 0.05, "valid": 0.05}

dirs = [
    name for name in os.listdir(dataset)
    if os.path.isdir(os.path.join(dataset, name))
]

outs = {
    "train": open(os.path.join(dataset, 'train.splits'), 'w'),
    "small_train": open(os.path.join(dataset, 'small_train.splits'), 'w'),
    "test": open(os.path.join(dataset, 'test.splits'), 'w'),
    "valid": open(os.path.join(dataset, 'valid.splits'), 'w')
}

for dir_name in dirs:
    files = [
        os.path.join(dir_name, name)
        for name in os.listdir(os.path.join(dataset, dir_name))
        if os.path.isfile(os.path.join(dataset, dir_name, name))
    ]
    num_files = len(files)
    last_index = 0

    print(files)

    for key, val in proportions.iteritems():
        upper = int(last_index + num_files * val)
        outs[key].write('\n'.join(files[last_index:upper]))
        outs[key].write('\n')
        last_index = upper

    outs["small_train"].write('\n'.join(files[0:int(num_files * 0.1)]))
    outs["small_train"].write('\n')
