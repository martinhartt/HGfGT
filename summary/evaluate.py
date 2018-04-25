from rouge import Rouge
import numpy
import argparse
import spacy

nlp = spacy.load('en_core_web_md')

parser = argparse.ArgumentParser(description='Evaluate the results of a model.')

parser.add_argument('-predictedSentsFile', default='',
          help='File containing the predicted summaries.')
parser.add_argument('-actualSentsFile', default='',
          help='File containing the actual summaries.')

opt = parser.parse_args()

def calculateRouge(predictedSents, actualSents):
    rouge = Rouge()
    scores = []

    for i in range(len(predictedSents)):
        scores.append(rouge.get_scores(predictedSents, actualSents))

    return scores

def calculateSemanticSimilarity(predictedSents, actualSents):
    scores = []
    for i in range(len(predictedSents)):
        predicted = nlp(predictedSents[i])
        actual = nlp(actualSents[i])
        score = predicted.similarity(actual)
        scores.append(score)

    return scores, numpy.mean(scores), numpy.median(scores)

def extractSents(fileName):
    return open(fileName).read().strip().split('\n')

def main():
    actualSents = extractSents(opt.actualSentsFile)
    predictedSents = extractSents(opt.predictedSentsFile)

    print(actualSents, predictedSents)

    print('Calculating Word2Vec similarity')
    semantic_scores, semantic_mean, semantic_median = calculateSemanticSimilarity(predictedSents, actualSents)
    print(semantic_scores, semantic_mean, semantic_median)
    print()

    print('Calculating ROUGE Score')
    rouge_scores = calculateRouge(predictedSents, actualSents)
    print(rouge_scores)
    print()

main()
