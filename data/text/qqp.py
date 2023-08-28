import os
from typing import Tuple, Any, List, Dict

from datasets import Dataset

from .glue import GLUE
from src.nlps.data import HFData, data_register, TaskType, DatasetSplitType
from src.nlps.data.data import ALL_DATASET_SPLIT
from src.nlps.utils.utils import SequencePairBatch


@data_register
class QQP(GLUE):
    _config_name = "qqp"
    _task_type = TaskType.CLASSIFICATION
    _dataset_used_portion = 1

    @property
    def label_column_name(self):
        return "label"

    @property
    def input_column_name(self):
        return "question1", 'question2'

    @property
    def input_name(self) -> str:
        return "question pair"

    @property
    def an_sample(self) -> Tuple[Any]:
        return ("How do I control my horny emotions?", "How do you control your horniness?"), 1

    def _load_dataset(self, splits: Tuple[DatasetSplitType] = ALL_DATASET_SPLIT):
        dataset: Dict[DatasetSplitType, Dataset] = super()._load_dataset(splits)
        for t, d in dataset.items():
            length = round(len(d) * self._dataset_used_portion)
            dataset[t] = Dataset.from_dict(d[:length])
        return dataset

    def extract_input_label_from_samples(self, samples: Dataset, *args, **kwargs):
        input_, label = super().extract_input_label_from_samples(samples, *args, **kwargs)
        if not isinstance(input_, SequencePairBatch):
            input_ = SequencePairBatch(*input_)
        return input_, label

    def _class_names(self) -> List[str]:
        return ["duplicate", "non-duplicate"]


@data_register
class QQPFG(QQP):
    @property
    def label_column_name(self):
        return "question2"

    @property
    def input_column_name(self):
        return "question1"

    _abbreviation = "qqpfg"
    _metric_name_path = "bleu"
    _task_type = TaskType.GENERATION

    def _load_dataset(self, splits: Tuple[DatasetSplitType] = ALL_DATASET_SPLIT):
        dataset: Dict[DatasetSplitType, Dataset] = super(QQPFG, self)._load_dataset(splits)
        for t, d in dataset.items():
            new_d = []
            for i in range(len(d)):
                if d[i]["label"] == 0:
                    continue
                new_d.append(d[i])
            new_d = Dataset.from_list(new_d)
            new_d = new_d.rename_column(original_column_name="label", new_column_name="duplication")
            dataset[t] = new_d

        return dataset

    @property
    def an_sample(self) -> Tuple[Any]:
        return "How do I control my horny emotions?", "How do you control your horniness?"

    def extract_input_label_from_samples(self, samples: Dataset, *args, **kwargs):
        return samples["question1"], samples["question2"]

    def _compute_metrics(self, predictions, labels, *args, **kwargs):
        predictions = [p.split() for p in predictions]
        references = [[l.split()] for l in labels]
        _metric = self._load_metric()
        result = _metric.compute(predictions=predictions, references=references)
        return {"bleu": result["bleu"]}

