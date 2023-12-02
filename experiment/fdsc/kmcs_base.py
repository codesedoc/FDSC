import inspect
import json
import os.path
import random
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import field, dataclass
from enum import Enum
from json import JSONEncoder
from typing import Optional, List, Any, Dict, Union, Iterable, Iterator, Tuple, Callable, Set

import numpy as np
import torch
from datasets import Dataset
from pytorch_metric_learning.losses import NTXentLoss
from torch import nn
from torch.nn import ModuleDict

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, AutoConfig, \
    AutoModelForSequenceClassification, DefaultDataCollator, DataCollatorWithPadding, Trainer, \
    PreTrainedTokenizer, PreTrainedTokenizerFast, PretrainedConfig, PreTrainedModel

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutput, Seq2SeqLMOutput

from src.nlps.data import TextData


class KMCSConfig(PretrainedConfig):

    def __init__(self, *args, **kwargs):
        self.with_contrastive_loss: bool = kwargs.pop('with_contrastive_loss', None)
        self.with_sentiment_loss: bool = kwargs.pop('with_sentiment_loss', None)
        self.sentiment_transferor_hidden_dim: int = kwargs.pop('sentiment_transferor_hidden_dim', 512)
        self.sent_token_location: int = kwargs.pop('sent_token_location', None)
        self.cont_token_location: int = kwargs.pop('cont_token_location', None)
        super().__init__(*args, **kwargs)


class KMCSTokenizer(PreTrainedTokenizer):
    pass


class KMCSTokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = KMCSTokenizer

    def _preprocess_input_sequence(self, sequence: str) -> str:
        return sequence

    def preprocess_input_sequence(self, sequence):
        if isinstance(sequence, list):
            return [self._preprocess_input_sequence(s) for s in sequence]
        elif isinstance(sequence, str):
            return self._preprocess_input_sequence(sequence)
        else:
            raise ValueError


class KMCSModelForPTR(PreTrainedModel):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sentiment_classifier = nn.Linear(self.hidden_state_dim_size, 2)
        self.sentiment_loss_fn = nn.BCELoss()
        config: KMCSConfig = self.config

        self.sentiment_transferor = nn.ModuleList([
                nn.Linear(self.hidden_state_dim_size, config.sentiment_transferor_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.sentiment_transferor_hidden_dim, self.hidden_state_dim_size)
            ])
        self.transfer_loss_fn = nn.MSELoss()
        self.backbone_encoder.register_forward_hook(self._encoder_forward_hock)
        self._runtime: Dict[str, Dict[str, Any]] = {
            "encoder_output": None
        }

        self.is_transfer_senti_hidden_state = True

        self._sentiment_output: Dict[str, np.ndarray] = None
        self.reset_sentiment_output()

    @property
    def sentiment_prediction(self) -> Optional[np.ndarray]:
        return self._sentiment_output["prediction"]

    @property
    def sentiment_label(self) -> Optional[np.ndarray]:
        return self._sentiment_output["label"]

    def reset_sentiment_output(self):
        self._sentiment_output = {
            "prediction": None,
            "label": None
        }

    @property
    @abstractmethod
    def backbone_encoder(self) -> nn.Module:
        pass

    def _encoder_forward_hock(self, encoder, encoder_input, encoder_output: BaseModelOutput):
        config: KMCSConfig = self.config
        if config.with_sentiment_loss:
            assert isinstance(config.sent_token_location, int)
            senti_state = encoder_output.last_hidden_state[:, config.sent_token_location, :]

            if self.is_transfer_senti_hidden_state:
                transferred_senti_state = senti_state
                for l in self.sentiment_transferor:
                    transferred_senti_state = l(transferred_senti_state)

                last_hidden_state = torch.cat([
                    encoder_output.last_hidden_state[:, :config.sent_token_location, :],
                    torch.unsqueeze(transferred_senti_state, dim=1),
                    encoder_output.last_hidden_state[:, config.sent_token_location+1:, :]
                ], dim=1)
                encoder_output.last_hidden_state = last_hidden_state
                encoder_output["transferred_senti_state"] = transferred_senti_state
            encoder_output["senti_hidden_state"] = senti_state

        if config.with_contrastive_loss:
            assert isinstance(config.cont_token_location, int)
            content_state = encoder_output.last_hidden_state[:, config.cont_token_location, :]
            encoder_output["content_state"] = content_state
        self._runtime["encoder_output"] = encoder_output
        return encoder_output

    @property
    @abstractmethod
    def hidden_state_dim_size(self):
        pass

    def forward(self, input_ids=None, attention_mask=None,  labels=None, label_input_ids=None, label_attention_mask=None, decoder_attention_mask=None,
                **kwargs):
        config: KMCSConfig = self.config

        self.is_transfer_senti_hidden_state = config.with_sentiment_loss

        output: Seq2SeqLMOutput = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                                  decoder_attention_mask=decoder_attention_mask, **kwargs)

        assert self._runtime["encoder_output"] is not None
        encoder_output = self._runtime["encoder_output"]

        if isinstance(output.loss, torch.Tensor):

            assert label_attention_mask is not None
            label_for_label = labels.clone().detach()

            self.is_transfer_senti_hidden_state = False
            output_for_label: Seq2SeqLMOutput = super().forward(input_ids=label_input_ids, attention_mask=label_attention_mask, labels=label_for_label,
                                                       **kwargs)
            loss_count = 2

            self.is_transfer_senti_hidden_state = config.with_sentiment_loss
            assert self._runtime["encoder_output"] is not None
            encoder_outputs_for_label = self._runtime["encoder_output"]
            c_loss = sentiment_loss= transfer_loss = 0
            if config.with_sentiment_loss:
                senti_hidden_state1 = encoder_output["senti_hidden_state"]
                transferred_senti_state1 = encoder_output["transferred_senti_state"]
                senti_hidden_state2 = encoder_outputs_for_label["senti_hidden_state"]
                senti_logits = self.sentiment_classifier(torch.cat([senti_hidden_state1, senti_hidden_state2], dim=0))
                batch_size = senti_logits.size(0) // 2
                senti_label = torch.nn.functional.one_hot(
                    torch.cat([torch.zeros(batch_size, dtype=torch.int), torch.ones(batch_size, dtype=torch.int64)],
                              dim=0).to(device=senti_logits.device),
                    num_classes = 2
                ).to(dtype=senti_logits.dtype)
                sentiment_pred = nn.Softmax(dim=-1)(senti_logits)
                sentiment_loss = self.sentiment_loss_fn(sentiment_pred, senti_label)
                transfer_loss = self.transfer_loss_fn(transferred_senti_state1, senti_hidden_state2)
                loss_count += 2

                # breakpoint()
                # senti_output = self._sentiment_output
                #
                # pred = np.argmax(sentiment_pred.detach().cpu().numpy(), axis=-1).astype(int)
                # labe = np.argmax(senti_label.detach().cpu().numpy(), axis=-1).astype(int)
                # if senti_output["prediction"] is None:
                #     senti_output["prediction"] = pred
                # else:
                #     senti_output["prediction"] = np.append(senti_output["prediction"], pred)
                # if senti_output["label"] is None:
                #     senti_output["label"] = labe
                # else:
                #     senti_output["label"] = np.append(senti_output["label"], labe)

                # if not self.training:
                #     with torch.no_grad():
                #         # breakpoint()
                #         senti_output = self._sentiment_output
                #         pred = np.argmax(nn.Softmax(dim=-1)(self.sentiment_classifier(transferred_senti_state1)).detach().cpu().numpy(), axis=-1).astype(int)
                #         labe = torch.ones(pred.shape, dtype=torch.int).detach().cpu().numpy().astype(int)
                #         if senti_output["prediction"] is None:
                #             senti_output["prediction"] = pred
                #         else:
                #             senti_output["prediction"] = np.append(senti_output["prediction"], pred)
                #         if senti_output["label"] is None:
                #             senti_output["label"] = labe
                #         else:
                #             senti_output["label"] = np.append(senti_output["label"], labe)

            if config.with_contrastive_loss:
                contrastive_loss = ContrastiveLoss()
                c_loss = contrastive_loss.compute(encoder_output["content_state"], encoder_outputs_for_label["content_state"])
                loss_count += 1

            output.loss = output.loss + output_for_label.loss + sentiment_loss + transfer_loss + c_loss

        return output


class ContrastiveLoss:
    def __init__(self):
        self.batch_size = None
        self._loss = None

    @property
    def loss(self):
        if self._loss is None:
            self._loss = NTXentLoss()
        return self._loss

    def _create_contrastive_pos_location(self, batch_size):
        return torch.cat(
            [
                torch.cat([
                    torch.zeros(batch_size, batch_size).byte(),
                    torch.diag(torch.ones(batch_size)).byte()
                ], dim=1),
                torch.cat([
                    torch.diag(torch.ones(batch_size)).byte(),
                    torch.zeros(batch_size, batch_size).byte()
                ], dim=1)
            ],
        )

    def compute(self, seq_embeddings, seq_embeddings_shadow):
        loss_func = self.loss
        # breakpoint()
        batch_size = seq_embeddings.size(dim=0)
        p_location = self._create_contrastive_pos_location(batch_size)
        n_location = 1 - p_location
        n_location.fill_diagonal_(0)

        assert torch.sum(p_location).item() == batch_size * 2
        assert torch.sum(n_location).item() == (batch_size * 2) * ((batch_size * 2) - 2)

        p_location = p_location.to(device=seq_embeddings.device)
        n_location = n_location.to(device=seq_embeddings.device)
        embeddings = torch.cat([seq_embeddings, seq_embeddings_shadow])

        indices_tuple = list(torch.where(p_location))
        indices_tuple.extend(torch.where(n_location))
        indices_tuple = tuple(indices_tuple)
        # print(f"*******************{embeddings.shape}****************************")
        try:
            loss = loss_func(embeddings, indices_tuple=indices_tuple)
        except Exception:
            print(f"*******************{embeddings.shape}****************************")
            raise

        return loss