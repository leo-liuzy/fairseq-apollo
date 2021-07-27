# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Unsupervised Cross-lingual Representation Learning at Scale
"""

from fairseq.models import register_model, register_model_architecture


from .hub_interface import RobertaHubInterface
from .model import RobertaModel


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


# class XLMRXCLEncoder(RobertaEncoder):
#     def __init__(self, args, dictionary):
#         super().__init__(dictionary)
#         self.args = args
#
#         if args.encoder_layers_to_keep:
#             args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
#
#     def forward(self, src_tokens, features_only=False, return_all_hiddens=False, masked_tokens=None, **unused):
#         """
#         Args:
#             src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
#             features_only (bool, optional): skip LM head and just return
#                 features. If True, the output will be of shape
#                 `(batch, src_len, embed_dim)`.
#             return_all_hiddens (bool, optional): also return all of the
#                 intermediate hidden states (default: False).
#
#         Returns:
#             tuple:
#                 - the LM output of shape `(batch, src_len, vocab)`
#                 - a dictionary of additional data, where 'inner_states'
#                   is a list of hidden states. Note that the hidden
#                   states have shape `(src_len, batch, vocab)`.
#         """
#         x, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
#         if not features_only:
#             x = self.output_layer(x, masked_tokens=masked_tokens)
#         return x, extra
#
#     def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
#         inner_states, _ = self.sentence_encoder(
#             src_tokens,
#             last_state_only=not return_all_hiddens,
#         )
#         features = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C
#         return features, {'inner_states': inner_states if return_all_hiddens else None}


@register_model('xlmr_xcl')
class XLMRXCLModel(XLMRModel):

    def __init__(self, args, encoder):
        super().__init__(args, encoder)
        # TODO (Leo): add pooler
        # self.pooler =

    def forward(self, src_tokens,
                src_tokens_mlm=None, src_lengths=None,  # set to None for subclassing
                tgt_tokens=None, tgt_tokens_mlm=None, tgt_lengths=None,
                features_only=False, return_all_hiddens=False,
                classification_head_name=None, **kwargs):
        # TODO (Leo): heavily(?) modify this function
        # TODO (Leo): I suspect dataset is not yet completed; mono corpus should be the first batch
        # TODO (Leo): 
        assert not (src_tokens_mlm is None or src_lengths is None), "src_tokens_mlm or src_lengths missing in batch"
        if classification_head_name is not None:
            features_only = True
        # self.encoder.sentence
        x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra


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