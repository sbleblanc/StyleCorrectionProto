import spacy
import fastBPE

class SpacyBPEPreprocess(object):

    def __init__(self,
                 bpe_codes_fn: str,
                 bpe_vocab_fn: str):
        self.nlp = spacy.load('en_core_web_sm',
                         disable=['tagger', 'parser', 'ner', 'entity_linker', 'textcat', 'entity_ruler', 'sentencizer',
                                  'merge_noun_chunks', 'merge_entities', 'merge_subtokens'])
        self.bpe = fastBPE.fastBPE(bpe_codes_fn, bpe_vocab_fn)

    def __call__(self, sentence: str):
        sentence = ' '.join([t.text for t in self.nlp(sentence)])
        sentence = self.bpe.apply([sentence])[0]
        sentence = sentence.lower()
        return sentence
