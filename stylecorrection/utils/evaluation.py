import nltk.translate as nt
from nltk.translate import nist_score
from typing import List

def compute_scores(predictions: List[List[str]],
                   references: List[List[str]]):
    scores = dict()

    #bleu
    scores['bleu'] = nt.bleu_score.corpus_bleu([references], predictions, smoothing_function=nt.bleu_score.SmoothingFunction().method4)

    #meteor
    # scores['meteor'] = 0.
    # for i in range(len(predictions)):
    #     scores['meteor'] += nt.meteor_score.single_meteor_score(' '.join(references[i]), ' '.join(predictions[i]))
    # scores['meteor'] /= len(predictions)

    #nist
    scores['nist'] = nist_score.corpus_nist([references], predictions)

    #ribs
    scores['ribes'] = nt.ribes_score.corpus_ribes([references], predictions)

    return scores