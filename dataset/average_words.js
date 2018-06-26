// Copyright (c) 2018, Martin Hartt

const { mean } = require('lodash');
const { readFileSync } = require('fs');

const ids = ['CAE', 'CPE', 'FCE', 'KET', 'PET']
const raw = ids.map(name => readFileSync(`${name}.all.article.txt`).toString())
const lengths = raw.map(file => file.split('\n').map(l => l.split(' ').length).filter(a => a > 1))

const averages = lengths.map(mean)

console.log(zipObject(ids, lengths.map(averages)))
