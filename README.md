# Headline Generation for General Text


This repository contains the reimplementations of neural headline generation systems built as part of my Advanced Computer Science MPhil dissertation at the University of Cambridge. The baseline is a PyTorch port of the [Lua implementation](https://github.com/facebookarchive/NAMAS) of the Neural Attention Model for Abstractive Summarization ([Rush et al. 2015](http://static.ijcai.org/proceedings-2017/0574.pdf)). This is extended with an implementation of the coarse-to-fine heirarchal attention system based on ([Tan et al. 2017](http://static.ijcai.org/proceedings-2017/0574.pdf)).

## Project abstract

> Existing efforts in the task of abstractive headline generation have limited their evaluation to the domain of newspaper articles. In this project I explore the effectiveness of neural approaches to abstractive headline generation on general text. I reimplement two neural abstractive text summarisers using the Pytorch library. As a baseline, a feed-forward attention-based neural network (Rush et al., 2015) is reimplemented. I implement an extension which features a coarse-to-fine approach, where extractive summarisers are first employed to find important sentences. These are used to train a recurrent neural network to predict the summary (Tan et al., 2017). Additionally, I utilise the OpenNMT framework (Klein et al., 2017) to find the effect of using Recurrent Neural Networks without the coarse-to-fine approach. Along with the Gigaword dataset featuring newspaper articles, the systems are evaluated using short stories from English Language exams. The style of this dataset is less journalistic, which highlights how well these systems perform on general text. Quantitative evaluation is conducted which measures the lexical and semantic similarity of the predicted and actual titles. I also evaluate the outputs qualitatively by measuring the grammaticality of the outputs. From the results, it is found that the OpenNMT (Klein et al., 2017) model trained to produce summaries from lead sentences, is able to produce the most accurate headlines. By evaluating the systems on a dataset with general text, it is found that they do not produce the same accuracies as the Gigaword dataset, suggesting they don’t effectively generalise across domains.


## Usage

1. Include the raw Gigaword dataset (with uncompressed files) under the path `data/agiga`.
2. Create train/test/validation splits with `bash bin/create_splits.sh data/agiga`
3. 
