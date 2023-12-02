from transformers import BartConfig, BartTokenizer, BartTokenizerFast, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.bart.modeling_bart import BartEncoder, BartForConditionalGeneration

from .kmcs_base import KMCSConfig, KMCSTokenizer, KMCSTokenizerFast, KMCSModelForPTR


class BartKMCSConfig(KMCSConfig, BartConfig):
    model_type = 'ims-bart'
    pass


class BartKMCSTokenizer(KMCSTokenizer, BartTokenizer):
    pass


class BartKMCSTokenizerFast(KMCSTokenizerFast, BartTokenizerFast):
    slow_tokenizer_class = BartKMCSTokenizer


class BartKMCSModelForPTR(KMCSModelForPTR, BartForConditionalGeneration):
    @property
    def hidden_state_dim_size(self):
        config: BartKMCSConfig = self.config
        return config.d_model

    config_class = BartKMCSConfig

    @property
    def backbone_encoder(self):
        return self.model.encoder

    def __init__(self, config: BartKMCSConfig):
        super().__init__(config)

AutoConfig.register(BartKMCSConfig.model_type, BartKMCSConfig)
AutoTokenizer.register(BartKMCSConfig, BartKMCSTokenizer, BartKMCSTokenizerFast)
AutoModelForSeq2SeqLM.register(BartKMCSConfig, BartKMCSModelForPTR)