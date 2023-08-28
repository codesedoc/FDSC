import os.path
from dataclasses import field
from enum import Enum
from typing import Optional, List, Any, Dict, Union, Iterator, Tuple, Callable, Set, OrderedDict

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import Sampler
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, T5Tokenizer

from .kmcs_t5 import T5KMCSConfig, T5KMCSTokenizerFast, T5KMCSModelForPTR
from experiment.utils import tuning_hp_prepare_stpg
from ..utils.ppf_utils import PGSTModelArgument, PTRModelArgument, Name2Backbone
from src.nlps.approach import Transformer, approach_register, TransformerArgument
from src.nlps.approach.transformer import DumpCallback, TransformerModelArgument
from src.nlps.argument import Argument, argument_class, ArgumentPool
from src.nlps.data import Name2DataClass, Data, merge_datasets, DatasetSplitType, TextData
from src.nlps.data.data import MultiTaskDataset, ALL_DATASET_SPLIT
from src.nlps.utils.runtime import RunTime
from .kmcs_base import KMCSConfig, KMCSTokenizer, KMCSTokenizerFast, KMCSModelForPTR
from .kmcs_bart import BartKMCSTokenizerFast, BartKMCSConfig, BartKMCSModelForPTR


class KMCSTrainer(Seq2SeqTrainer):
    pass

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     for callback in self.callback_handler.callbacks:
    #         if isinstance(callback, DumpCallback):
    #             config: KMCSConfig = self.model.config
    #             callback.path_suffix_factory = lambda: config.dynamic_taskcase.name


class KMCSVariantType(Enum):
    SELF = "self"
    WITHOUT_CONTR = "woc"
    WITHOUT_SENTI = "wos"
    WITHOUT_SENTI_CONTR = "wosc"
    BASELINE = "baseline"

    def __str__(self):
        return self.value


@argument_class
class KMCSArgument(TransformerArgument):
    variant: str = field(
        default="self",
        metadata={
            'help': "Specify one of the three experiment settings",
            'choices': str([e.value for e in KMCSVariantType])
        }
    )

    pos_token: str = field(
        default="[POS]",
        metadata={
            'help': "Specify the special token for extracting the positive feature of text",
        }
    )

    neg_token: str = field(
        default="[NEG]",
        metadata={
            'help': "Specify the special token for extracting the negative feature of text",
        }
    )

    cont_token: str = field(
        default="[CON]",
        metadata={
            'help': "Specify the special token for extracting the content feature of text",
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.variant:KMCSVariantType = KMCSVariantType(self.variant)
        self._all_special_tokens = [self.neg_token, self.pos_token, self.cont_token]
        self.senti_token_location = None
        self.cont_token_location = None

    @property
    def special_tokens(self):
        result = {t: None for t in self._all_special_tokens}
        if self.variant in [KMCSVariantType.WITHOUT_CONTR, KMCSVariantType.WITHOUT_SENTI_CONTR]:
            result.pop(self.cont_token)
        if self.variant in [KMCSVariantType.WITHOUT_SENTI, KMCSVariantType.WITHOUT_SENTI_CONTR]:
            result.pop(self.neg_token)
            result.pop(self.pos_token)
        return result

    @property
    def if_with_content(self):
        return self.cont_token in self.special_tokens

    @property
    def if_with_sentiment(self):
        return self.neg_token in self.special_tokens


@argument_class
class KMCSModelArgument(PTRModelArgument):
    pass


class KMCSDataCollator(DataCollatorForSeq2Seq):

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:

        if "labels" in features[0]:
            assert 'label_attention_mask' in features[0]
            label_input_ids = [i.get("labels") for i in features]
            label_attention_mask = [i.pop("label_attention_mask") for i in features]
            batch = super().__call__(features, return_tensors)
            batch_tmp = super().__call__([{
                    "input_ids": i,
                    "attention_mask": a,
                    "labels": i,
                } for i, a in zip(label_input_ids, label_attention_mask)
            ], return_tensors)
            batch["label_input_ids"] = batch_tmp["input_ids"]
            batch["label_attention_mask"] = batch_tmp["attention_mask"]
        else:
            batch = super().__call__(features, return_tensors)
        return batch


class MultiTaskSampler(Sampler[int]):
    def __init__(self, sampler: Sampler[int], batch_size: int, dataset: Dataset,
                 task_names: Set[str]) -> None:
        super().__init__(dataset)
        self._sampler = sampler
        self._batch_size = batch_size
        self._task_names = task_names
        self._dataset = dataset

    def __iter__(self) -> Iterator[List[int]]:
        final_index_order = []
        sub_sampler: Dict[str, list] = {n: [] for n in self._task_names}
        data_source: Dataset = self._dataset
        for i in iter(self._sampler):
            task_name = data_source[i]['task']
            sub_sampler[task_name].append(i)
            if len(sub_sampler[task_name]) == self._batch_size:
                final_index_order.extend(sub_sampler[task_name])
                sub_sampler[task_name] = []
        for i in final_index_order:
            yield i

    def __len__(self) -> int:
        return len(self._dataset)


Name2BackboneClass = {
    "bart": {
        "config": BartKMCSConfig,
        "tokenizer": BartKMCSTokenizerFast,
        "model": BartKMCSModelForPTR
    },
    "t5": {
        "config": T5KMCSConfig,
        "tokenizer": T5KMCSTokenizerFast,
        "model": T5KMCSModelForPTR
    }
}

@approach_register
class KeepMeaningChangeStyle(Transformer):
    _abbreviation = 'kmcs'
    _training_arg_class = Seq2SeqTrainingArguments
    _argument_class = KMCSArgument
    _trainer_class = KMCSTrainer
    _data_collator_class = KMCSDataCollator
    _model_argument_class = KMCSModelArgument

    _auxiliary_name2taskcase = {

    }
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    @property
    def check_point(self):
        checkpoint = super().check_point
        if not os.path.isdir(checkpoint):
            checkpoint = self.model_args.backbone
        return checkpoint

    def _tokenizer_init(self, *args, **kwargs):
        tokenizer: KMCSTokenizer = super()._tokenizer_init(*args, **kwargs)
        argument: KMCSArgument = self.args
        additional_tokens = []
        if argument.if_with_sentiment:
            additional_tokens.extend([argument.neg_token, argument.pos_token])
        if argument.if_with_content:
            additional_tokens.append(argument.cont_token)

        additional_tokens = list(set(additional_tokens) - set(tokenizer.vocab.keys()))
        tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens})
        return tokenizer

    def _config_init(self, *args, **kwargs):
        config: KMCSConfig = super()._config_init(*args, **kwargs)
        argument: KMCSArgument = self.args
        data: TextData = self.processing_data
        num_additional_tokens = len(self.tokenizer.additional_special_tokens)
        config.max_length = min(data.max_length + num_additional_tokens, 1024)
        config.min_length = max(data.min_length + num_additional_tokens, 2)
        config.with_contrastive_loss = argument.if_with_content
        config.with_sentiment_loss = argument.if_with_sentiment
        config.sent_token_location = argument.senti_token_location
        config.cont_token_location = argument.cont_token_location
        return config

    def _model_init(self, *args, **kwargs):
        transformer: KMCSModelForPTR = super()._model_init(*args, **kwargs)
        argument: KMCSArgument = self.args
        if len(argument.special_tokens) > 0:
            transformer.resize_token_embeddings(len(self.tokenizer))
        return transformer

    def _prediction_to_output(self, prediction: np.ndarray, runtime: Optional[Dict[str, Any]] = None) -> Union[
        np.ndarray, List]:
        result = self.tokenizer.batch_decode(prediction, skip_special_tokens = True)
        return result

    def release_model(self, model=None):
        super().release_model(model)

    @property
    def auto_classes(self):
        model_args: PGSTModelArgument = self.model_args
        return Name2BackboneClass[model_args.backbone_name]

    def _request_datasets(self, data: TextData = None, *args, **kwargs) -> Optional[Tuple[Dataset]]:
        arguments: KMCSArgument = self.args
        data = data if isinstance(data, TextData) else self.precessing_data
        input_name = data.input_column_name
        label_name = data.label_column_name
        prefix_components_of_input_seq = []
        prefix_components_of_label_seq = []
        if arguments.if_with_sentiment:
            prefix_components_of_input_seq.append(arguments.neg_token)
            prefix_components_of_label_seq.append(arguments.pos_token)
        if arguments.if_with_content:
            prefix_components_of_input_seq.append(arguments.cont_token)
            prefix_components_of_label_seq.append(arguments.cont_token)

        def add_prefix_components(old_datset, prefix_components, column_name):
            if len(prefix_components) <= 0:
                return old_datset

            new_column = list(map(lambda s: " ".join([*prefix_components, s]), old_datset[column_name]))
            return old_datset.remove_columns(column_name).add_column(column_name, new_column)

        for split, dataset in zip(self._process_data_split, data.dataset(self._process_data_split)):
            new_dataset = add_prefix_components(dataset, prefix_components_of_input_seq, input_name)
            data.set_dataset(split, add_prefix_components(new_dataset, prefix_components_of_label_seq, label_name))

        len(self.tokenizer.additional_special_tokens)
        return super()._request_datasets(data, *args, **kwargs)

    def pre_tokenization_call_back(self, sequence):
        tokenizer: KMCSTokenizerFast = self.tokenizer
        return tokenizer.preprocess_input_sequence(sequence)

    def _preprocess(self, samples: Union[Dataset, Dict[str, Any]], *args, **kwargs):
        from src.nlps.data import TextData
        data: TextData = self.precessing_data
        kwargs["max_length"] = data.max_length + len(self.tokenizer.additional_special_tokens)
        result = super()._preprocess(samples, *args, **kwargs)

        input_ids = result["input_ids"]
        min_length = min([len(i) for i in input_ids])
        input_ids = np.array([i[:min_length] for i in input_ids], dtype=int)
        argument: KMCSArgument = self.args
        if argument.if_with_sentiment:
            neg_locations = np.where(input_ids == self.tokenizer.convert_tokens_to_ids(argument.neg_token))[1]
            pos_locations = np.where(input_ids == self.tokenizer.convert_tokens_to_ids(argument.pos_token))[1]
            # breakpoint()
            assert not (len(neg_locations) > 0 and len(pos_locations) > 0)
            senti_locations = neg_locations if len(neg_locations) > 0 else pos_locations
            sent_token_location = int(senti_locations[0])
            assert np.all(senti_locations == sent_token_location)
            if argument.senti_token_location is None:
                argument.senti_token_location = sent_token_location
            else:
                assert argument.senti_token_location == sent_token_location
        else:
            argument.senti_token_location = None

        if argument.if_with_content:
            cont_locations = np.where(input_ids == self.tokenizer.convert_tokens_to_ids(argument.cont_token))[1]
            cont_token_location = int(cont_locations[0])
            assert np.all(cont_locations == cont_token_location)
            if argument.cont_token_location is None:
                argument.cont_token_location = cont_token_location
            else:
                assert argument.cont_token_location == cont_token_location
        else:
            argument.cont_token_location = None

        return result

    def _process(self, *args, **kwargs):
        super()._process(*args, **kwargs)

    @classmethod
    def collect_argument(cls):
        super().collect_argument()
        Argument.update_defaults_for_fields(cls.training_arg_class, {
            # 'evaluation_strategy': "epoch",
            # 'learning_rate': 3e-5,
            'save_total_limit': 3,
            'weight_decay': 0.01,
            'predict_with_generate': True,
            'use_mps_device': False
        })

    @property
    def tune_prepare(self):
        result = super().tune_prepare
        result = tuning_hp_prepare_stpg(self, result)
        return result

    @property
    def trail_name(self):
        args: KMCSArgument = self.args
        return f'{self.abbreviation}_{args.dataset}_{args.variant.value}'

    @property
    def path_component(self):
        args: KMCSArgument = self.args
        return f'{super().path_component}/{args.variant.value}'

    def _visulization_with_matplotlib(self, name=None, output_dir: str = None, show: bool = False, f_size=(5, 4),
                                      **kwargs):
        def _ax_meta_info(tokenier_, sequences):
            tokenized_sequence = tokenier_(sequences)["input_ids"]
            sequence_lengths = np.array([len(s) for s in tokenized_sequence], dtype=int)
            length2count = {}
            for l in sequence_lengths:
                if l not in length2count:
                    length2count[l] = 0
                length2count[l] += 1
            average_length = round(sequence_lengths.sum() / len(tokenized_sequence), 1)
            return {"sequence_lengths":sequence_lengths, "length2count": length2count, "average_length": average_length}

        name2tokenizer: Dict[str, KMCSTokenizerFast] = {
            "BART": BartKMCSTokenizerFast.from_pretrained(Name2Backbone["bart"]),
            "T5": BartKMCSTokenizerFast.from_pretrained(Name2Backbone["t5"])
        }

        data: TextData = self.processing_data
        all_sequence = data.all_text

        name2ax_meta_info = {}
        for n, t in name2tokenizer.items():
            meta_info = None
            for s, sequence in all_sequence.items():
                info = _ax_meta_info(t, sequence)
                if meta_info is None:
                    meta_info = info.copy()
                else:
                    length2count = {}
                    length_set = set(meta_info["length2count"])
                    length_set.update(set(info["length2count"]))
                    for l in length_set:
                        length2count[l] = meta_info["length2count"].get(l, 0) + info["length2count"].get(l, 0)
                    meta_info["length2count"] = length2count
                    meta_info["average_length"] += info["average_length"]
                    np.append(meta_info["sequence_lengths"], info["sequence_lengths"])
                meta_info[s] = info

            meta_info["average_length"] = round(meta_info["average_length"] / len(all_sequence), 1)
            length_count_pairs = sorted(meta_info["length2count"].items(), key=lambda x: x[0])
            x_axis = np.array(list(p[0] for p in length_count_pairs), dtype=int)
            y_axis = np.array(list(p[1] for p in length_count_pairs), dtype=int)
            name2ax_meta_info[n] = meta_info
            name2ax_meta_info[n]["x_axis"] = x_axis
            name2ax_meta_info[n]["y_axis"] = y_axis
            name2ax_meta_info[n]["name"] = n
        import matplotlib
        from matplotlib import pyplot as plt
        font = {'family': 'Times',
                'weight': 'bold',
                'size': 12}

        matplotlib.rc('font', **font)
        matplotlib.rc('xtick', labelsize=16)
        # matplotlib.rc('ytick', labelsize=20)

        num_axs = 2
        f_size = f_size[0] * num_axs, f_size[1]
        fig, axs = plt.subplots(1, num_axs, figsize=f_size)
        labels = []
        lengths = []
        for n, info in name2ax_meta_info.items():
            lengths.append(info["sequence_lengths"].tolist())
            labels.append(n)

        bplot = axs[0].boxplot( lengths,
                        vert=True,  # vertical box alignment
                        patch_artist=True,  # fill with color
                        labels=labels, # will be used to label x-ticks
                        autorange=True,
                        showfliers = False,
                        showmeans = True
                        )
        axs[0].set_title('Box Plot of the Length Tokenized by PLMs', fontsize=16)
        axs[0].set_ylabel('Length', fontsize=14)
        # fill with colors
        colors = ['lightblue', 'pink']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        splits = DatasetSplitType.TRAIN, DatasetSplitType.VALIDATION, DatasetSplitType.TEST
        x = np.arange(len(splits))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        for n, info in name2ax_meta_info.items():
            offset = width * multiplier
            rects = axs[-1].bar(x + offset, [info[s]["average_length"] for s in splits], width, align="edge", label=n)
            axs[-1].bar_label(rects, padding=2)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axs[-1].set_title('Average Length on Datset Splits', fontsize=16)
        axs[-1].set_ylabel('Length', fontsize=14)
        xticks = [s.value for s in splits]
        xticks[1] = "development"
        axs[-1].set_xticks(x + width, xticks)
        axs[-1].set_ylim(0,40)
        axs[-1].legend()

        output_dir = "tmp/visualization"
        path = os.path.join(output_dir, f"{data.abbreviation}_statistic")

        path_dir = os.path.abspath(os.path.dirname(path))
        if not os.path.exists(path_dir):
            os.system(f"mkdir -p {path_dir}")
        plt.savefig(f"{path}.png", dpi=120, format="png")
        plt.savefig(f"{path}.svg", dpi=120, format="svg")
        show = True
        if show:
            plt.show()
