# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Unsupervised Cross-lingual Representation Learning at Scale
"""

import torch
from typing import List
from torch import nn
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from .hub_interface import RobertaHubInterface
from .model import RobertaModel, RobertaEncoder


@register_model('xlmr')
class XLMRModel(RobertaModel):

    @classmethod
    def hub_models(cls):
        return {
            'xlmr.base': 'http://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz',
            'xlmr.large': 'http://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.tar.gz',
        }

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='sentencepiece', **kwargs):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return RobertaHubInterface(x['args'], x['task'], x['models'][0])


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, args):
        super().__init__()
        self.pooler_type = args.pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "cls_after_pooler",
                                    "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type
        if self.pooler_type in ["cls_after_pooler"]:
            self.dense = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
            self.activation = nn.Tanh()

    def forward(self, attention_mask: torch.tensor, hidden_states: List[torch.tensor]):
        # pooler_output = outputs.pooler_output
        # hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return hidden_states[-1][0]
        elif self.pooler_type in ['cls_after_pooler']:
            first_token_tensor = hidden_states[-1][0]
            pooled_output = self.dense(first_token_tensor)
            pooled_output = self.activation(pooled_output)
            return pooled_output
        elif self.pooler_type == "avg":
            return ((hidden_states[-1] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


@register_model('xlmr_xcl')
class XLMRXCLModel(XLMRModel):

    def __init__(self, args, encoder):
        super().__init__(args, encoder)
        # TODO (Leo): add pooler
        self.pooler = Pooler(args)
        self.pooler_requring_all_hiddens = ["avg_top2", "avg_first_last"]

    @staticmethod
    def add_args(parser):
        XLMRModel.add_args(parser)
        parser.add_argument('--pooler-type', default="cls", type=str,
                            choices=["cls", "cls_before_pooler", "cls_after_pooler",
                                     "avg", "avg_top2", "avg_first_last"],
                            help='probability of replacing a token with mask')

    def forward(self, src_tokens,
                src_positions=None,  # set to None for subclassing
                force_positions=True,
                features_only=False, return_all_hiddens=False,
                classification_head_name=None, **kwargs):
        """
        Depends on different task, src_tokens could means different things.

        For MLM, src_tokens is the masked sequence
        For Contrastive Learning, it could means unmasked sequences
        For TLM, src_tokens is the masked and concatenated sequences

        Similar situation for src_positions.
        """
        return super().forward(src_tokens,
                               src_positions=src_positions,
                               force_positions=force_positions,
                               features_only=features_only,
                               return_all_hiddens=return_all_hiddens,
                               classification_head_name=classification_head_name, **kwargs)


@register_model_architecture('xlmr_xcl', 'xlmr_xcl_base')
def base_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    # (Leo): xlmr use learned embedding
    args.dropout = getattr(args, 'dropout', 0.1)  # (Leo): this includes embedding dropout
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
    args.encoder_layers_to_keep = getattr(args, 'encoder_layers_to_keep', None)
    args.encoder_layerdrop = getattr(args, 'encoder_layerdrop', 0.0)

    args.pooler_type = getattr(args, "pooler_type", "cls")