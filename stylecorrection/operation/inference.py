import os
from stylecorrection.operation.common import *
from stylecorrection.loaders.corpus import *
from stylecorrection.models.transformer import TransformerS2S


class InferenceOperation(Operation):

    @classmethod
    def infer(cls,
              cl: StreamingH5CorpusLoader,
              line: str,
              model: TransformerS2S,
              beam_width: int,
              max_len_scale: int,
              noising_beta: float,
              temperature: float,
              device: str):

        encoded = cl.encode_sentence(line).to(device)

        beam_decoded = model.beam_decode_3(
            encoded,
            torch.tensor([cl.bos_idx], dtype=torch.long).to(device),
            beam_width=beam_width,
            max_len=int(encoded.shape[0] * max_len_scale),
            end_token=cl.eos_idx,
            noising_beta=noising_beta,
            temperature=temperature,
            top_only=False,
            device=device
        )

        return cl.decode_tensor(beam_decoded[0])[0]

    def __init__(self,
                 global_conf: GlobalConfig,
                 inference_conf: InferenceConfig,
                 model_conf: TransformerConfig,
                 preprocess_conf: PreprocessConfig,
                 device: str):
        super(InferenceOperation, self).__init__(
            global_conf,
            device,
            model_conf=model_conf,
            hd5_dataset_config=inference_conf.hd5_dataset
        )
        self.inference_conf = inference_conf
        self.preprocess_conf = preprocess_conf

        if inference_conf.force_cpu:
            self.device = 'cpu'

        if inference_conf.preprocess:
            from stylecorrection.utils.preprocess import SpacyBPEPreprocess
            self.spacy_bpe_pp = SpacyBPEPreprocess.from_conf(preprocess_conf)

        self.load_dataset(inference_conf.max_len)
        pretrained_mdl_path = os.path.expandvars(inference_conf.pretrained_model)
        self.load_model(len(self.vocab), pretrained_fn=pretrained_mdl_path)

        self.source_input_fn = os.path.expandvars(inference_conf.source_fn)
        self.hyp_output_fn = os.path.expandvars(inference_conf.hyp_fn)
        self.buffering = -1 if inference_conf.output_buffering else 1

    def _infer(self, line: str):
        return self.infer(
            self.cl_train,
            line,
            self.model,
            self.inference_conf.beam_width,
            self.inference_conf.max_len_scale,
            self.inference_conf.noising_beta,
            self.inference_conf.temperature,
            self.device
        )

    def run(self):
        with open(self.source_input_fn, 'r') as in_f:
            with open(self.hyp_output_fn, 'w', buffering=self.buffering) as out_f:
                for li, line in enumerate(in_f):
                    if li < self.config['inference']['line_offset']:
                        continue
                    line = line.strip()
                    if self.config['inference']['preprocess']:
                        line = self.spacy_bpe_pp(line)
                    print('IN  : {}'.format(line))
                    if 0 < self.inference_conf.max_len < len(line.split(' ')):
                        print('TOO LONG')
                        continue

                    decoded_sentence = self._infer(line)

                    if self.inference_conf.remove_bpe_placeholder:
                        decoded_sentence = decoded_sentence.replace("@@ ", "")
                    if self.inference_conf.output_parallel:
                        out_f.write('{} <split> {}\n'.format(line, decoded_sentence))
                    else:
                        out_f.write('{}\n'.format(decoded_sentence))
                    print('OUT : {}'.format(decoded_sentence))
