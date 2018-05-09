from rouge import Rouge
import numpy
import argparse
import spacy

nlp = spacy.load('en_core_web_lg')

parser = argparse.ArgumentParser(
    description='Evaluate the results of a model.')

parser.add_argument(
    '-predictedSentsFile',
    default='',
    help='File containing the predicted summaries.')
parser.add_argument(
    '-actualSentsFile',
    default='',
    help='File containing the actual summaries.')

opt = parser.parse_args()


def calculateRouge(predictedSents, actualSents):
    rouge = Rouge()

    return rouge.get_scores(predictedSents, actualSents)


def calculateSemanticSimilarity(predictedSents, actualSents):
    scores = []
    for i in range(len(predictedSents)):
        predicted = nlp(predictedSents[i])
        actual = nlp(actualSents[i])
        score = predicted.similarity(actual)
        scores.append(score)

    return scores, numpy.mean(scores), numpy.median(scores)


def extractSents(fileName):
    return unicode(open(fileName).read(), 'utf8').strip().split('\n')


def main():
    print('Calculating Word2Vec similarity...')
    actualSents = extractSents(opt.actualSentsFile)
    predictedSents = extractSents(opt.predictedSentsFile)

    semantic_scores, semantic_mean, semantic_median = calculateSemanticSimilarity(
        predictedSents, actualSents)

    print('Calculating ROUGE Score...')
    rouge_scores = calculateRouge(predictedSents, actualSents)

    pad = 10
    delim = ' | '

    print(delim.join([
        str("#").ljust(3),
        str("Rouge 1").ljust(pad),
        str("Rouge 2").ljust(pad),
        str("Rouge L").ljust(pad),
        str("Semantic sim.").ljust(pad)
    ]))

    print('=' * (3 + (pad + 3) * 4))

    for i in range(len(predictedSents)):
        print(delim.join([
            str(i).ljust(3),
            str(rouge_scores[i]['rouge-1']['f']).ljust(pad)[:pad],
            str(rouge_scores[i]['rouge-2']['f']).ljust(pad)[:pad],
            str(rouge_scores[i]['rouge-l']['f']).ljust(pad)[:pad],
            str(semantic_scores[i]).ljust(pad)[:pad]
        ]))


main()
