import copy
import pickle
import random

from datasets import Dataset

from src.nlps.data import Data, data_register, DatasetSplitType, GeneralDataset, DataDirCategory, DataContainer, DataWrapper, \
    TaskType
from typing import Dict, Union, List, Tuple, Any
import os
import pandas as pd


@data_register
class ASPSReview(Data):
    _task_type = TaskType.GENERATION

    def _preprocess(self, *args, **kwargs):
        pass

    _abbreviation = 'asap_review'

    class ReviewNode(DataWrapper):
        def __init__(self, *args, next_node=None, **kwargs):
            super().__init__(kwargs['content'])
            self.next = next_node
            self.id = kwargs['id']
            if self.content is None:
                self.need_reframe = True
                self.source_text = kwargs['source_text']
            else:
                self.need_reframe = False
                self.source_text = self.content

    class Review:
        def __init__(self, _id, content, labels=None):
            self.id = _id
            self.content = content
            self.labels = labels
            self.reframe_nodes: List[ASPSReview.ReviewNode] = list()

        @property
        def reframe(self):
            sequences = ' '.join([r.content for r in self.reframe_nodes])
            return sequences

    def _dataset_for_application(self, runtime: Dict[str, Any], *args, **kwargs) -> Dataset:
        result: Dict[str, List] = DataContainer(['input', 'output_wrapper'])
        test = kwargs.get("smoke_test", False)
        raw_dir = self.data_dir(category=DataDirCategory.RAW)
        name = 'test' if test else 'review_with_aspect'
        raw_path = os.path.join(raw_dir, 'aspect_data', f'{name}.jsonl')
        if not os.path.isfile(raw_path):
            raise ValueError

        df = pd.read_json(raw_path, lines=True)
        dn = df.to_numpy()
        review_dict: Dict[str, List[ASPSReview.Review]] = dict()
        for item in dn:
            _id, text, labels = item
            review = ASPSReview.Review(_id=_id, content=copy.deepcopy(text), labels=labels)
            index = 0
            review_nodes = []
            for l in labels:
                start, end, aspect = l
                start, end = int(start), int(end)
                if index != start:
                    review_nodes.append(ASPSReview.ReviewNode(id=_id, content=text[index: start]))
                content = text[start: end]
                if 'negative' in aspect:
                    result['input'].append(content)
                    node = ASPSReview.ReviewNode(id=_id, content=None, source_text=content)
                    result['output_wrapper'].append(node)
                else:
                    node = ASPSReview.ReviewNode(id=_id, content=content)
                review_nodes.append(node)
                index = end+1
            if index != len(text):
                review_nodes.append(ASPSReview.ReviewNode(id=_id, content=text[index: len(text)]))

            for i, n in enumerate(review_nodes[:-1]):
                n.next = review_nodes[i+1]

            review.reframe_nodes = review_nodes
            if not _id in review_dict:
                review_dict[_id] = list()
            review_dict[_id].append(review)
        result = Dataset.from_dict(result)
        runtime["review_dict"] = review_dict
        return result

    def _output_reframed_review_plain(self, application_dir, review_dict):
        output_file = os.path.join(application_dir, f'{self.abbreviation}.csv')
        save_data = {
            'id': [],
            'source_review': [],
            'reframed_review': []
        }
        for _id, reviews in review_dict.items():
            for i, r in enumerate(reviews):
                save_data['id'].append(f'{_id}_r{i}')
                save_data['source_review'].append(r.content)
                save_data['reframed_review'].append(r.reframe)

        df = pd.DataFrame.from_dict(save_data)
        df.to_csv(output_file, index=False)

    def _output_reframed_review_markdown(self, application_dir, review_dict):
        colors = ('#fcf080', '#80dfdc', '#98fb98', '#d1d1d1', '#ffa500', '#809aeb', '#ee82ee', '#7b68ee', '#0ec4ba', '#006666')
        annotation = '- [ ] Paraphrase\n\t- [ ] Need Previous Context\n\t- [ ] Need Following Context\n\t- [x] Confused\n' \
                     '- [ ] Positive Reframing\n\t- [ ] Need Previous Context\n\t- [ ] Need Following Context\n\t- [x] Confused'
        file_index = 0
        paper_index = 0
        paper_limitation = 10
        file_limitation = 1
        review_count = 0
        output_lines = []
        keys = list(review_dict.keys())
        random.seed(2022)
        random.shuffle(keys)
        assert len(set(keys)) == len(review_dict.keys())
        for _id in keys:
            reviews = review_dict[_id]
            print(_id)
            for r_i, r in enumerate(reviews):
                color_index = 0
                source_sentences = []
                reframe_sentences = []
                if _id == 'ICLR_2018_633':
                    _id = _id

                for node in r.reframe_nodes:
                    if node.need_reframe:
                        source = node.source_text.replace('`', r'\`').replace('#', r'\#')
                        reframe = node.content.replace('`',r'\`').replace('#',r'\#')
                        source_sentences.append(f'<font style="background: {colors[color_index % len(colors)]}">{source}</font>')
                        reframe_sentences.append(f'<font style="background: {colors[color_index % len(colors)]}">{reframe}</font>')
                        color_index += 1
                    else:
                        source_sentences.append(node.source_text.replace('`',r'\`').replace('#',r'\#'))
                        reframe_sentences.append(node.content.replace('`',r'\`').replace('#',r'\#'))

                if color_index > 0:
                    review_count += 1
                    output_lines.append(f'# {review_count}. {_id}_r{r_i}')
                    output_lines.append('## Source Review')
                    output_lines.append(' '.join(source_sentences))
                    output_lines.append('## Reframed Review')
                    output_lines.append(' '.join(reframe_sentences))
                    output_lines.append('## Annotation')
                    for c_i in range(color_index):
                        output_lines.append(f'### Sequence Pair <font style="background: {colors[c_i % len(colors)]}">{c_i+1}</font>')
                        output_lines.append(annotation)
            if len(output_lines) > 0:
                paper_index += 1
                if paper_index % paper_limitation == 0:
                    file_index += 1
                    output_file = os.path.join(application_dir, f'{self.abbreviation}_{file_index}.md')
                    with open(output_file, 'w') as f:
                        for item in output_lines:
                            f.write("%s\n\n" % item)
                    output_lines = []
                    review_count = 0
                    if file_index == file_limitation:
                        break

        if len(output_lines) > 0:
            file_index += 1
            output_file = os.path.join(application_dir, f'{self.abbreviation}_{file_index}.md')
            with open(output_file, 'w') as f:
                for item in output_lines:
                    f.write("%s\n\n" % item)
            output_lines = []

    def _application_finish_call_back(self, runtime: Dict[str, Any]):
        output_wrapper = runtime.get("dataset")["output_wrapper"]
        output = runtime.get("dataset")["output"]
        assert len(output_wrapper) == len(output)
        for o, o_w in zip(output, output_wrapper):
            o_w.content = o
        # cls._output_reframed_review_plain(application_dir)
        output_dir = runtime.get("output_dir")
        render_dir = os.path.join(output_dir, 'render')
        if os.path.exists(render_dir):
            os.system(f'rm -r {render_dir}')
        os.system(f'mkdir -p {render_dir}')
        self._output_reframed_review_markdown(render_dir, runtime.get("review_dict"))
