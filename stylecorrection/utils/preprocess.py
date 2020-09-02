import spacy
import fastBPE
import os
from stylecorrection.utils.config import PreprocessConfig

class SpacyBPEPreprocess(object):

    @classmethod
    def from_conf(cls, preprocess_conf: PreprocessConfig):
        codes_fn = os.path.expandvars(preprocess_conf.bpe_codes_fn)
        bpe_vocab_fn = os.path.expandvars(preprocess_conf.bpe_vocab_fn)
        return cls(codes_fn, bpe_vocab_fn, preprocess_conf.output_lower)

    def __init__(self,
                 bpe_codes_fn: str,
                 bpe_vocab_fn: str,
                 output_lower: bool):
        self.nlp = spacy.load('en_core_web_sm',
                         disable=['tagger', 'parser', 'ner', 'entity_linker', 'textcat', 'entity_ruler', 'sentencizer',
                                  'merge_noun_chunks', 'merge_entities', 'merge_subtokens'])
        self.bpe = fastBPE.fastBPE(bpe_codes_fn, bpe_vocab_fn)
        self.output_lower = output_lower

    def __call__(self, sentence: str):
        sentence = ' '.join([t.text for t in self.nlp(sentence)])
        sentence = self.bpe.apply([sentence])[0]
        if self.output_lower:
            sentence = sentence.lower()
        return sentence
