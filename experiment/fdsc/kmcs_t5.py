from transformers import T5Config, T5Tokenizer, T5TokenizerFast, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.t5.modeling_t5 import T5Stack, T5ForConditionalGeneration

from .kmcs_base import KMCSConfig, KMCSTokenizer, KMCSTokenizerFast, KMCSModelForPTR


class T5KMCSConfig(KMCSConfig, T5Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = False

    model_type = 'kmcs-t5'
    pass


class T5KMCSTokenizer(KMCSTokenizer, T5Tokenizer):
    pass



class T5KMCSTokenizerFast(KMCSTokenizerFast, T5TokenizerFast):
    slow_tokenizer_class = T5KMCSTokenizer

    def _preprocess_input_sequence(self, sequence: str) -> str:
        sequence = super()._preprocess_input_sequence(sequence)
        prefix = "summarize: "
        return prefix + sequence



class T5KMCSModelForPTR(KMCSModelForPTR, T5ForConditionalGeneration):
    @property
    def backbone_encoder(self):
        return self.encoder

    config_class = T5KMCSConfig

    @property
    def hidden_state_dim_size(self):
        config: T5KMCSConfig = self.config
        return config.d_model

AutoConfig.register(T5KMCSConfig.model_type, T5KMCSConfig)
AutoTokenizer.register(T5KMCSConfig, T5KMCSTokenizer, T5KMCSTokenizerFast)
AutoModelForSeq2SeqLM.register(T5KMCSConfig, T5KMCSModelForPTR)
