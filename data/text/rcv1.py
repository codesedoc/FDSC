import dataclasses
import json
import math
import os
import pickle
import random
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Any, List, Dict, Optional, Iterable, Callable
import xml.etree.ElementTree as ET

import pandas
from datasets import Dataset
from pandas import DataFrame
from tqdm import tqdm

from experiment.data.metrics.classification_metric.classification_metric import ClassificationMetric, ClassificationType
from src.nlps.argument import DataArgument
from src.nlps.data import Data, DatasetSplitType
from src.nlps.data.data import ALL_DATASET_SPLIT, DataDirCategory, data_register, TaskType
from src.nlps.data.hierarchical_data import HierarchicalData, Hierarchy, Node, HierarchyInfo, \
    create_meta_hierarchy_class


@dataclass
class RCV1HierarchyInfo(HierarchyInfo):
    code2description: Optional[Dict[str, str]] = None

    @property
    def root_code(self) -> Optional[str]:
        result = None
        root = self.root
        if isinstance(root, Node):
            result = root.id
        return result

class RCV1Hierarchy(Hierarchy):
    @property
    def info(self) -> Optional[RCV1HierarchyInfo]:
        info = super().info
        if not isinstance(info, HierarchyInfo):
            return None
        result: RCV1HierarchyInfo = RCV1HierarchyInfo(info)
        nodes: Iterable[Node] = result.nodes
        result.code2description = {n.id: n.meta_info['description'] for n in nodes}
        return result


RCV1MetaHierarchy = create_meta_hierarchy_class(RCV1Hierarchy)


class RCV1SplitMode(Enum):
    LYRL2004 = 'lyrl2004'
    SIMPLE = 'simple'


@dataclass
class RCV1Argument(DataArgument):
    split_assign_mode: Optional[str] = dataclasses.field(
        default='lyrl2004',
        metadata={
            'help': "Specify the mode of method to assigning dataset splits.",
            'choices': "['lyrl2004', 'simple']"
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.split_assign_mode = RCV1SplitMode(self.split_assign_mode)


@data_register
class RCV1(HierarchicalData):
    _abbreviation = 'rcv1'
    _metric_name_path = 'experiment/data/metrics/classification_metric'
    _task_type = TaskType.CLASSIFICATION
    _argument_class = RCV1Argument

    @property
    def input_column_name(self) -> str:
        return "title"

    @property
    def label_column_name(self) -> str:
        return "category_codes"

    @property
    def _field_name2value_factory(self) -> Optional[Dict[str, Callable]]:
        return {
            'id': int,
            self.label_column_name: lambda item: eval(item),
            'description': lambda item: eval(item),
        }

    @property
    def label_id_column_name(self) -> str:
        return "category_codes"

    def label_id2samples(self, dataset: Dataset) -> Dict[str, List]:
        if not isinstance(dataset, Dataset):
            return None

        label_id2samples: Dict[str, List] = dict()

        for i, s in enumerate(tqdm(dataset, desc="Extract labels")):
            label_ids = eval(s[self.label_id_column_name])
            for l_i in label_ids:
                assert isinstance(l_i, str)
                if l_i not in label_id2samples:
                    label_id2samples[l_i] = list()
                label_id2samples[l_i].append(s)

        return label_id2samples

    @classmethod
    def _init_meta_hierarchy(cls) -> RCV1MetaHierarchy:
        hierarchy_path = os.path.join(cls.data_dir(category=DataDirCategory.RAW), 'category', 'rcv1.topics.hier.orig.txt')
        codes_path = os.path.join(cls.data_dir(category=DataDirCategory.RAW), 'category', 'rcv1.topics.txt')
        code_set = {'None', 'Root'}
        with open(codes_path, mode='r') as f:
            rows = f.readlines()
        for r in rows:
            r = r.strip()
            assert r not in code_set
            code_set.add(r)

        arcs = list()
        with open(hierarchy_path, mode='r') as f:
            rows = f.readlines()
        for r in rows:
            items = r.strip().split('child-description:')
            assert len(items) == 2
            node_items = items[0].split()
            assert len(node_items) == 4
            assert node_items[0] == 'parent:' and node_items[2] == 'child:'
            assert node_items[1] in code_set and node_items[3] in code_set
            arcs.append({
                'parent': node_items[1] if node_items[1] != 'None' else None,
                'child': node_items[3],
                'info': items[1]
            })
        result: RCV1MetaHierarchy = RCV1MetaHierarchy(cls.abbreviation)
        code2node: Dict[str, Node] = dict()
        code2parent_code: Dict[str, str] = dict()
        for a in arcs:
            code = a['child']
            assert code not in code2node
            node = Node(_id=a['child'], name=code)
            node.meta_info['description'] = a['info']
            if a['parent'] is None:
                assert result.is_empty
                result.root = node
            code2node[code] = node
            code2parent_code[a['child']] = a['parent']
        assert not result.is_empty
        for c, n in code2node.items():
            if n is result.root:
                continue
            parent = code2node[code2parent_code[c]]
            parent.add_child(n)
            assert n.parent == parent
        assert len(result.info.nodes) == len(code2node)
        return result

    @property
    def an_sample(self) -> Tuple[Any]:
        return None

    def _parse_raw_data(self) -> Dict[str, str]:
        meta_hierarchy_info: RCV1HierarchyInfo = self.meta_hierarchy.info
        text_dir = os.path.join(self.raw_dir, 'text')
        text_paths = [n for n in os.listdir(text_dir) if
                      os.path.isfile(os.path.join(text_dir, n)) and os.path.splitext(n)[0].isnumeric()]
        assert len(text_paths) == 365
        samples: List[Dict[str, Any]] = list()
        broken_samples: List[Dict[str, Any]] = list()
        pbar = tqdm(text_paths)
        for t_p in pbar:
            pbar.set_description(f"Parsing {t_p}:")
            t_p = os.path.join(text_dir, t_p)
            assert zipfile.is_zipfile(t_p)
            with zipfile.ZipFile(t_p) as zf:
                info_list = zf.infolist()
                for i in info_list:
                    content = []
                    assert not i.is_dir() and i.filename.endswith('.xml')
                    with zf.open(i.filename, mode='r') as f:
                        for line in f:
                            try:
                                line = line.decode(encoding='utf-8')
                            except UnicodeDecodeError:
                                try:
                                    line = line.decode(encoding='cp437')
                                except UnicodeDecodeError:
                                    line = line.decode(encoding='utf-8', errors='backslashreplace')
                            content.append(line)
                    content = ''.join(content)
                    xml_root = ET.fromstring(content)

                    sample = {
                        'id': -1,
                        "title": None,
                        "category_codes": [],
                        "description": [],
                        "date": None,
                        "filename": i.filename,
                    }
                    if xml_root.tag == "newsitem":
                        sample['id'] = int(xml_root.attrib["itemid"])
                        sample['date'] = xml_root.attrib["date"]

                    for child in xml_root:
                        if child.tag == "title":
                            assert sample['title'] is None
                            sample['title'] = child.text
                        if child.tag == "metadata":
                            metadata = child
                            for item in metadata:
                                if item.tag == "codes" and item.attrib == {"class": "bip:topics:1.0"}:
                                    assert len(sample['category_codes']) == 0
                                    sample['category_codes'] = [str(code.attrib["code"]) for code in item]
                                    try:
                                        sample['description'] = [meta_hierarchy_info.code2description[code]
                                                                 for code in sample['category_codes']]
                                    except Exception:
                                        raise

                    try:
                        assert isinstance(sample['id'], int) and sample['id'] != -1 and\
                               isinstance(sample['title'], str) and \
                               isinstance(sample['category_codes'], list) and \
                               isinstance(sample['date'], str)
                        samples.append(sample)
                    except AssertionError:
                        if sample['title'] is None:
                            broken_samples.append(sample)
                        else:
                            raise
        return samples, broken_samples

    @staticmethod
    def _simple_split(samples: List[Dict]) -> Dict[DatasetSplitType, DataFrame]:
        random.seed(1)
        random.shuffle(samples)
        total = len(samples)
        split_samples = {}
        start_index = 0
        split_rate = {
            DatasetSplitType.TRAIN: 0.7,
            DatasetSplitType.VALIDATION: 0.2,
            DatasetSplitType.TEST: 0.1,
        }
        for s, r in split_rate:
            end_index = start_index + round(r * total)
            split_samples[s] = samples[start_index: end_index]
            start_index = end_index
        assert start_index == total
        return split_samples

    @staticmethod
    def _lyrl2004_split(split_id_dir, samples: List[Dict]) -> Dict[DatasetSplitType, DataFrame]:
        train_file = 'lyrl2004_tokens_train.dat'
        test_files = [
            'lyrl2004_tokens_test_pt0.dat', 'lyrl2004_tokens_test_pt1.dat',
            'lyrl2004_tokens_test_pt2.dat', 'lyrl2004_tokens_test_pt3.dat'
        ]

        def _extract_id(file_path) -> list[int]:
            file_path = os.path.join(split_id_dir, file_path)
            _ids = []
            with open(file_path) as f:
                for l in f:
                    if l.strip().startswith('.I'):
                        items = l.split()
                        assert len(items) == 2
                        _ids.append(int(items[1]))
            return _ids

        def _load_ids():
            lyrl2004_ids_file = 'lyrl2004_ids.json'
            file_path = os.path.join(split_id_dir, lyrl2004_ids_file)
            if os.path.isfile(file_path):
                with open(file_path, mode='r') as f:
                    lyrl2004_ids = json.load(f)
            else:
                lyrl2004_ids = OrderedDict()
                lyrl2004_ids['Split Mode'] = 'LYRL2004'
                # lyrl2004_ids['Number of Total Ids'] = -1
                # lyrl2004_ids['Number of Train Ids'] = -1
                # lyrl2004_ids['Number of Test Ids'] = -1
                lyrl2004_ids['train'] = _extract_id(train_file)
                _test_ids = []
                for tf in test_files:
                    _test_ids.extend(_extract_id(tf))
                lyrl2004_ids['test'] = _test_ids
                with open(file_path, mode='w') as f:
                    json.dump(lyrl2004_ids, f, indent=4)
            return lyrl2004_ids['train'], lyrl2004_ids['test']

        origin_train_ids, test_ids = _load_ids()

        split_samples = {
            DatasetSplitType.TRAIN: [], DatasetSplitType.VALIDATION: [], DatasetSplitType.TEST: [],
        }

        random.seed(1)
        dev_ids = random.sample(origin_train_ids, k=round(len(origin_train_ids) * 0.1))
        dev_ids_set = set(dev_ids)

        train_ids = []
        for i in origin_train_ids:
            if i not in dev_ids_set:
                train_ids.append(i)
        train_ids_set = set(train_ids)
        test_ids_set = set(test_ids)

        rest_samples = []
        for i, s in {int(s['id']): s for s in samples}.items():
            if i in train_ids_set:
                split_samples[DatasetSplitType.TRAIN].append(s)
            elif i in dev_ids_set:
                split_samples[DatasetSplitType.VALIDATION].append(s)
            elif i in test_ids_set:
                split_samples[DatasetSplitType.TEST].append(s)
            else:
                rest_samples.append(s)

        assert len(split_samples[DatasetSplitType.TRAIN]) + len(split_samples[DatasetSplitType.VALIDATION]) == 23149
        assert len(split_samples[DatasetSplitType.TEST]) == 781265
        return split_samples, rest_samples

    def _assign_splits(self, samples: List[Dict]) -> Dict[DatasetSplitType, DataFrame]:
        self.args: RCV1Argument
        if self.args.split_assign_mode == RCV1SplitMode.SIMPLE:
            result = self._simple_split(samples)
            rest_samples = []
        elif self.args.split_assign_mode == RCV1SplitMode.LYRL2004:
            result, rest_samples = self._lyrl2004_split(os.path.join(self.raw_dir, 'split'), samples)
        else:
            raise ValueError

        return result, rest_samples

    def _preprocess(self, split: DatasetSplitType, *args, **kwargs):
        origin_samples_path = os.path.join(self.raw_dir, 'origin_samples.bin')
        broken_samples_path = os.path.join(self.raw_dir, 'broken_samples.json')

        if self.args.force_preprocess or not os.path.isfile(origin_samples_path):
            samples, broken_samples = self._parse_raw_data()
            with open(origin_samples_path, mode='wb') as f:
                pickle.dump(samples, f)
            if len(broken_samples) > 0:
                with open(broken_samples_path, mode='w') as f:
                    json.dump({
                        'Data Name': self.abbreviation,
                        'Number of Broken Samples': len(broken_samples),
                        'Broken Samples': broken_samples
                    }, f)
        else:
            with open(origin_samples_path, mode='rb') as f:
                samples = pickle.load(f)
            print(f"Load parsed samples from archive file '{origin_samples_path}'")

        split_samples, rest_samples = self._assign_splits(samples)

        pandas.DataFrame(split_samples[split]).to_csv(self._preprocessed_files[split], index=False, quotechar='"')

        if len(rest_samples) > 0:
            rest_samples_path = os.path.join(self.data_dir(category=DataDirCategory.PREPROCESSED), f'{self.args.split_assign_mode.value}_rest_sampels.json')
            with open(rest_samples_path, mode='w') as f:
                json.dump({
                    'Data Name': self.abbreviation,
                    'Number of Rest Samples': len(rest_samples),
                    'Rest Samples': rest_samples
                }, f, indent=4)

    def _category_names(self) -> List[str]:
        info: RCV1HierarchyInfo = self.meta_hierarchy.info
        root_code = info.root_code
        return [c for c in info.code2description.keys() if c != root_code]

    def _load_metric(self):
        metric: ClassificationMetric = super()._load_metric()
        metric.classification_type = ClassificationType.MCML
        return metric


# if __name__ == "__main__":
#     rcv1: RCV1 = RCV1()
#     rcv1.meta_hierarchy.info.to_json('tmp/rcv1.json')
#     rcv1.dataset()
#     hierarchy = rcv1.hierarchy()
#
#     pass

distribution={'smdistributed': {
            'dataparallel': {
                'enabled': True
            }
        }}


