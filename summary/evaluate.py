from rouge import Rouge
import numpy
import argparse
import spacy
from collections import Counter
from benepar.spacy_plugin import BeneparComponent
from neuralcoref import Coref


coref = Coref()

nlp = spacy.load('en_core_web_lg')

# nlp.add_pipe(BeneparComponent('benepar_en'))

parser = argparse.ArgumentParser(
    description='Evaluate the results of a model.')

parser.add_argument(
    'inputFile',
    default='',
    help='File containing the generation outputs.')
parser.add_argument(
    '--csv',
    default=False,
    type=bool,
    help='Should output format be csv?')

opt = parser.parse_args()


def calculate_rouge(predicted_sents, actual_sents):
    rouge = Rouge()

    return rouge.get_scores(predicted_sents, actual_sents)


def calculate_semantic_similarity(predicted_sents, actual_sents):
    scores = []
    for i in range(len(predicted_sents)):
        predicted = nlp(predicted_sents[i])
        actual = nlp(actual_sents[i])
        score = predicted.similarity(actual)
        scores.append(score)

    return scores, numpy.mean(scores), numpy.median(scores)


def count_repetitions(sent):
    return sum([item[1] - 1 for item in Counter(sent.split()).items()])

def calculate_repetitions(predicted_sents):
    return [count_repetitions(sent) for sent in predicted_sents]

def coref_check(predicted_sents):
    clusters = coref.one_shot_coref(utterances=u"Mary is nice. She loves to bike.")
    print(clusters)

    mentions = coref.get_mentions()
    print(mentions)

    utterances = coref.get_utterances()
    print(utterances)

    resolved_utterance_text = coref.get_resolved_utterances()
    print(resolved_utterance_text)

def consituency_parse(predicted_sents):
    for sent in predicted_sents:
        doc = nlp(sent)
        sent = list(doc.sents)[0]
        print(sent._.parse_string)
        # (S (NP (NP (DT The) (NN time)) (PP (IN for) (NP (NN action)))) (VP (VBZ is) (ADVP (RB now))) (. .))
        print(sent._.labels)
        # ('S',)
        print(list(sent._.children)[0])
        # The time for action

def main():
    input = open(opt.inputFile).read().split("# Evaluating ")[1:]

    if opt.csv:
        print("Source,#,Rouge 1,Rouge 2,Rouge L,Semantic,Reps")

    for source_output in input:
        entries = [c for c in source_output.split("\n\n\n") if bool(c.strip())]
        source = entries[0]

        actual_sents = []
        predicted_sents = []

        for entry in entries[1:]:
            input, actual, output = [c[2:] for c in entry.split("\n") if c.strip() != '']

            actual_sents.append(unicode(actual, 'utf8'))
            predicted_sents.append(unicode(output, 'utf8'))


        semantic_scores, semantic_mean, semantic_median = calculate_semantic_similarity(
            predicted_sents, actual_sents)

        rouge_scores = calculate_rouge(predicted_sents, actual_sents)

        repetitions = calculate_repetitions(predicted_sents)

        # consituency_parse(predicted_sents)
        coref_check(predicted_sents)
        exit()

        if opt.csv:
            for i in range(len(predicted_sents)):
                print("{},{},{},{},{},{},{}".format(source, str(i), str(rouge_scores[i]['rouge-1']['f']), str(rouge_scores[i]['rouge-2']['f']), str(rouge_scores[i]['rouge-l']['f']), str(semantic_scores[i]), str(repetitions[i])))
        else:
            pad = 10
            delim = ' | '
            print("\n\n# {}\n".format(source))

            header = delim.join([
                str("#").ljust(3),
                str("Rouge 1").ljust(pad),
                str("Rouge 2").ljust(pad),
                str("Rouge L").ljust(pad),
                str("Semantic").ljust(pad),
                str("Reps").ljust(pad)
            ])
            print(header)

            print('=' * len(header))

            for i in range(len(predicted_sents)):
                print(delim.join([
                    str(i).ljust(3),
                    str(rouge_scores[i]['rouge-1']['f']).ljust(pad)[:pad],
                    str(rouge_scores[i]['rouge-2']['f']).ljust(pad)[:pad],
                    str(rouge_scores[i]['rouge-l']['f']).ljust(pad)[:pad],
                    str(semantic_scores[i]).ljust(pad)[:pad],
                    str(repetitions[i]).ljust(pad)[:pad]
                ]))

            print(delim.join([
                str("AVG"),
                str(avg([rouge_scores[i]['rouge-1']['f'] for i in range(len(predicted_sents))])).ljust(pad)[:pad],
                str(avg([rouge_scores[i]['rouge-2']['f'] for i in range(len(predicted_sents))])).ljust(pad)[:pad],
                str(avg([rouge_scores[i]['rouge-l']['f'] for i in range(len(predicted_sents))])).ljust(pad)[:pad],
                str(avg([semantic_scores[i] for i in range(len(predicted_sents))])).ljust(pad)[:pad],
                str(avg([repetitions[i] for i in range(len(predicted_sents))])).ljust(pad)[:pad]
            ]))

def avg(list):
    return sum(list) / float(len(list))

main()
