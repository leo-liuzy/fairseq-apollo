# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
import torch
import argparse
from collections import OrderedDict

from fairseq.data import (
    data_utils,
    Dictionary,
    encoders,
    ConcatDataset,
    ConcatSentencesDataset,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    PositionDataset,
    PrependTokenDataset,
    MultiCorpusSampledDataset,
    RawLabelDataset,
    RoundRobinZipDatasets,
    ResamplingDataset,
    SortDataset,
    TokenBlockDataset,
)
from fairseq.tasks import FairseqTask, register_task
from fairseq import utils


logger = logging.getLogger(__name__)


class round_robin_sampler_generator():
    def __init__(self, init_state=1):
        self.idx = init_state

    def __call__(self, x: list):
        self.idx += 1
        self.idx %= len(x)
        return x[self.idx - 1]


def uniform_sampler(x):
    # Sample from uniform distribution
    return np.random.choice(x, 1).item()


def sampler_mapping(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() == "round_robin":
        return round_robin_sampler_generator()
    elif s.lower() == "uniform":
        return uniform_sampler
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


@register_task('xlm_xcl')
class XlmrXcl(FairseqTask):
    """Task for training XLM (only MLM and TLM)
       and XCL (monolingual/translational contrastive learning)"""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--data', required=True,
                            help='colon separated path to monolingual data directories list, '
                                 'will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--use-mono-data', action="store_true",
                            help='colon separated path to monolingual data directories list, '
                                 'will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--use-para-data', action="store_true",
                            help='colon separated path to parallel data directories list,'
                                 ' will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--use-mcl', action="store_true",
                            help='Use Monolingual Contrastive Learning')
        parser.add_argument('--use-tcl', action="store_true",
                            help='Use Translational Contrastive Learning')
        parser.add_argument('--lang-batch-sampler', choices=["uniform", "round_robin"], type=sampler_mapping,
                            default="round_robin",
                            help='colon separated path to parallel data directories list,'
                                 ' will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--langs', default=None,
                            help='The list of languages we include. colon separated')
        parser.add_argument('--lang-pairs', default=None,
                            help='The list of language pairs we include. colon separated')
        parser.add_argument('--objective-use-mode', default='all',
                            choices=['all', 'alter'],
                            help='If set to "all", we add all objective together and jointly optimize.'
                                 'If "alter", alternate between mono and parallel objective.')
        parser.add_argument('--mlm-coeff', type=float, default=1.0,
                            help='Coefficient for Masked Language Model (MLM) loss')
        parser.add_argument('--tlm-coeff', type=float, default=1.0,
                            help='Coefficient for Translational Language Model (TLM) loss')
        parser.add_argument('--mcl-coeff', type=float, default=1.0,
                            help='Coefficient for Monolingual Contrastive Learning (MCL, i.e. SimCSE) loss')
        parser.add_argument('--tcl-coeff', type=float, default=1.0,
                            help='Coefficient for Translational Contrastive Learning (TCL) loss')
        parser.add_argument('--mono-sample-break-mode', default='complete',
                            choices=['none', 'complete', 'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--para-sample-break-mode', default='eos',
                            choices=['none', 'complete', 'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments '
                                 'per sample for BERT dataset')
        parser.add_argument('--mask-prob', default=0.15, type=float,
                            help='probability of replacing a token with mask')
        parser.add_argument('--leave-unmasked-prob', default=0.1, type=float,
                            help='probability that a masked token is unmasked')
        parser.add_argument('--random-token-prob', default=0.1, type=float,
                            help='probability of replacing a token with a random token')
        parser.add_argument('--freq-weighted-replacement', action='store_true',
                            help='sample random replacement words based on word frequencies')
        parser.add_argument('--mask-whole-words', default=False, action='store_true',
                            help='mask whole words; you may also want to set --bpe')
        parser.add_argument('--multilang-mono-sampling-alpha', type=float, default=1.0,
                            help='smoothing alpha for sample rations across multiple datasets')
        parser.add_argument('--multilang-para-sampling-alpha', type=float, default=1.0,
                            help='smoothing alpha for sample rations across multiple datasets')
        parser.add_argument('--temp-mcl', type=float, default=1.0,
                            help='Temperature used in Monolingual Contrastive Learning (MCL, i.e. SimCSE)')
        parser.add_argument('--temp-tcl', type=float, default=1.0,
                            help='Temperature used in Translational Contrastive Learning (TCL)')


    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed

        # add mask token
        self.mask_idx = dictionary.add_symbol('<mask>')
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        self.data_dir = paths[0]
        self.use_mono_data = args.use_mono_data
        self.use_para_data = args.use_para_data
        self.objective_use_mode = args.objective_use_mode
        assert self.use_mono_data or self.use_para_data

        self.mono_data = os.path.join(self.data_dir, "monolingual") if self.use_mono_data else None
        if self.use_mono_data:
            assert os.path.exists(self.mono_data)
        self.para_data = os.path.join(self.data_dir, "bilingual") if self.use_para_data else None
        if self.use_para_data:
            assert os.path.exists(self.para_data)
        self.languages = utils.split_paths(self.args.langs)
        self.language_pairs = utils.split_paths(self.args.lang_pairs)

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'))
        logger.info('dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def _get_whole_word_mask(self):
        # create masked input and targets
        if self.args.mask_whole_words:
            bpe = encoders.build_bpe(self.args)
            if bpe is not None:

                def is_beginning_of_word(i):
                    if i < self.source_dictionary.nspecial:
                        # special elements are always considered beginnings
                        return True
                    tok = self.source_dictionary[i]
                    if tok.startswith('madeupword'):
                        return True
                    try:
                        return bpe.is_beginning_of_word(tok)
                    except ValueError:
                        return True

                mask_whole_words = torch.ByteTensor(list(
                    map(is_beginning_of_word, range(len(self.source_dictionary)))
                ))
        else:
            mask_whole_words = None
        return mask_whole_words

    def _get_sample_prob(self, dataset_lens, sampling_alpha=1.0):
        """
        Get smoothed sampling porbability by languages. This helps low resource
        languages by upsampling them.
        """
        prob = dataset_lens / dataset_lens.sum()
        smoothed_prob = prob ** sampling_alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        self._load_mono_dataset(split, epoch, combine, **kwargs)
        if self.para_data:
            self._load_para_dataset(split, epoch, combine, **kwargs)

    def _load_xlm_dataset(self, split, split_path, lang1, lang2=None, combine=False, **kwargs):
        assert lang1 is not None
        # lang2 == None means TLM
        maybe_datasets = []  # len=1 if monolingual, len=2 if bilingual
        mask_whole_words = self._get_whole_word_mask()
        for lang in (lang1, lang2):
            if lang is None:
                continue
            dataset = data_utils.load_indexed_dataset(
                f"{split_path}.{lang}" if lang2 else split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            if dataset is None:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, f"{split_path}.{lang}"))

            # create continuous blocks of tokens
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode=self.args.para_sample_break_mode if lang2
                else self.args.mono_sample_break_mode,
            )
            logger.info('loaded {} blocks from: {}'.format(len(dataset), split_path))
            maybe_datasets.append(dataset)

        # monolingual dataset is a must
        src_tokens = PadDataset(
            ConcatDataset(maybe_datasets[:]),
            pad_idx=self.source_dictionary.pad(),
            left_pad=False,
        )
        # (Leo): this is static masking
        src_mlm_input_dataset, src_mlm_output_dataset = MaskTokensDataset.apply_mask(
            src_tokens,
            self.source_dictionary,  # TODO(Leo) : If we later decide to do BPE, we need to change this
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.args.seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
            freq_weighted_replacement=self.args.freq_weighted_replacement,
            mask_whole_words=mask_whole_words,
        )
        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        src_mlm_input_dataset = PrependTokenDataset(src_mlm_input_dataset, self.source_dictionary.bos())
        src_mlm_output_dataset = PrependTokenDataset(src_mlm_output_dataset, self.source_dictionary.bos())
        # (Leo): without wrapping the two dataset with PadDataset will cause collator to fail
        src_mlm_input_dataset = PadDataset(
            src_mlm_input_dataset,
            pad_idx=self.source_dictionary.pad(),
            left_pad=False,
        )
        src_mlm_output_dataset = PadDataset(
            src_mlm_output_dataset,
            pad_idx=self.source_dictionary.pad(),
            left_pad=False,
        )
        src_positions = PositionDataset(src_tokens, pad_idx=self.source_dictionary.pad())
        src_positions = PadDataset(
            src_positions, pad_idx=self.source_dictionary.pad(), left_pad=False,
        )
        # optionally create the same dataset for target language
        tgt_tokens, tgt_mlm_input_dataset, tgt_mlm_output_dataset, tgt_positions = None, None, None, None
        if lang2:
            tgt_tokens = PadDataset(
                ConcatDataset(maybe_datasets[::-1]),
                pad_idx=self.target_dictionary.pad(),
                left_pad=False,
            )
            tgt_mlm_input_dataset, tgt_mlm_output_dataset = MaskTokensDataset.apply_mask(
                tgt_tokens,
                self.target_dictionary,  # TODO(Leo): If we later decide to do BPE, we need to change this
                pad_idx=self.target_dictionary.pad(),
                mask_idx=self.mask_idx,
                seed=self.args.seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
                freq_weighted_replacement=self.args.freq_weighted_replacement,
                mask_whole_words=mask_whole_words,
            )
            tgt_mlm_input_dataset = PrependTokenDataset(tgt_mlm_input_dataset, self.source_dictionary.bos())
            tgt_mlm_output_dataset = PrependTokenDataset(tgt_mlm_output_dataset, self.source_dictionary.bos())
            tgt_mlm_input_dataset = PadDataset(
                tgt_mlm_input_dataset,
                pad_idx=self.target_dictionary.pad(),
                left_pad=False,
            )
            tgt_mlm_output_dataset = PadDataset(
                tgt_mlm_output_dataset,
                pad_idx=self.target_dictionary.pad(),
                left_pad=False,
            )
            tgt_positions = PositionDataset(tgt_tokens, pad_idx=self.target_dictionary.pad())
            tgt_positions = PadDataset(
                tgt_positions, pad_idx=self.target_dictionary.pad(), left_pad=False,
            )

        dummy_final_tokens = ConcatSentencesDataset(src_tokens, tgt_tokens) if lang2\
            else src_tokens
        final_target = ConcatSentencesDataset(src_mlm_output_dataset, tgt_mlm_output_dataset) if lang2\
            else src_mlm_output_dataset
        if lang2:
            # parallel corpus is optional
            lang_dataset = NestedDictionaryDataset(
                {
                    'net_input': {
                        'src_tokens': src_tokens,
                        'src_tokens_mlm': src_mlm_input_dataset,
                        'src_positions': src_positions,
                        'src_lengths': NumelDataset(src_tokens, reduce=False),

                        'tgt_tokens': tgt_tokens,
                        'tgt_tokens_mlm': tgt_mlm_input_dataset,
                        'tgt_positions': tgt_positions,
                        'tgt_lengths': NumelDataset(tgt_tokens, reduce=False),
                    },
                    'target': {
                        "src_mlm": PadDataset(
                            src_mlm_output_dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        "tgt_mlm": PadDataset(
                            tgt_mlm_output_dataset,
                            pad_idx=self.target_dictionary.pad(),
                            left_pad=False,
                        )
                    },
                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(dummy_final_tokens, reduce=True),
                    # 'lang_id': RawLabelDataset([lang_id] * src_dataset.sizes.shape[0]),
                },
                sizes=[dummy_final_tokens.sizes],
            )
        else:
            lang_dataset = NestedDictionaryDataset(
                {
                    'net_input': {
                        'src_tokens': src_tokens,
                        'src_tokens_mlm': src_mlm_input_dataset,
                        'src_positions': src_positions,
                        'src_lengths': NumelDataset(src_tokens, reduce=False),
                    },
                    'target': {
                        'src_mlm': PadDataset(
                            final_target,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                    },
                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(dummy_final_tokens, reduce=True),
                    # 'lang_id': RawLabelDataset([lang_id] * src_dataset.sizes.shape[0]),
                },
                sizes=[dummy_final_tokens.sizes],
            )
        return lang_dataset

    def _load_para_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = utils.split_paths(self.para_data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        language_pairs = sorted(
            name for name in self.language_pairs
            if os.path.isdir(os.path.join(data_path, name))
        )
        assert len(language_pairs) == len(self.language_pairs)
        logger.info("Training on {0} language pairs: {1}".format(len(language_pairs), language_pairs))
        logger.info("Language pair to id mapping: {0}".format({
            lang_pair: id for id, lang_pair in enumerate(language_pairs)
            })
        )

        lang_datasets = []
        for lang_pair_id, language_pair in enumerate(language_pairs):
            lang1, lang2 = language_pair.split('-')
            split_path = os.path.join(data_path, language_pair, split)
            lang_dataset = self._load_xlm_dataset(split, split_path, lang1=lang1, lang2=lang2,
                                                  combine=combine, **kwargs)
            lang_datasets.append(lang_dataset)

        dataset = self._maybe_resample_datasets(split, lang_datasets, lang_datasets_meta=language_pairs,
                                                sampling_alpha=self.args.multilang_para_sampling_alpha,
                                                epoch=epoch, combine=combine, **kwargs)

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))

        # we assume mono datasets is a must
        assert split in self.datasets
        para_final_dataset = SortDataset(
            dataset,
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )
        # Note that first is mono and para, since OrderedDict is used, such order is remembered
        self.datasets[split] = MultiCorpusSampledDataset(OrderedDict({
            "mono_" + split: self.datasets[split],
            "para_" + split: para_final_dataset
        }), sampling_func=self.args.lang_batch_sampler)

    def _maybe_resample_datasets(self, split, lang_datasets, lang_datasets_meta, sampling_alpha=1.0,
                                 epoch=1, combine=False, **kwargs):
        """
        Resample the mono/para datasets;
        if it is on validation split, we additionally save individual mono/para datasets

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )
        logger.info(
            'loaded total {} blocks for all languages'.format(
                dataset_lengths.sum(),
            )
        )
        if split == self.args.train_subset:
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(dataset_lengths,
                                                 sampling_alpha)
            logger.info("Sample probability by language (pair): ", {
                    lang: "{0:.4f}".format(sample_probs[id])
                    for id, lang in enumerate(lang_datasets_meta)
                }
            )
            size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
            logger.info("Up/Down Sampling ratio by language (pair): ", {
                    lang: "{0:.2f}".format(size_ratio[id])
                    for id, lang in enumerate(lang_datasets_meta)
                }
            )

            resampled_lang_datasets = [
                ResamplingDataset(
                    lang_datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(lang_datasets)
            ]
            dataset = ConcatDataset(resampled_lang_datasets)
        else:
            dataset = ConcatDataset(lang_datasets)
            lang_splits = [split]
            for lang_id, lang_dataset in enumerate(lang_datasets):
                split_name = split + '_' + lang_datasets_meta[lang_id]
                lang_splits.append(split_name)
                self.datasets[split_name] = lang_dataset

            # [TODO]: This is hacky for now to print validation ppl for each
            # language individually. Maybe need task API changes to allow it
            # in more generic ways.
            if split in self.args.valid_subset:
                self.args.valid_subset = self.args.valid_subset.replace(
                    split, ','.join(lang_splits)
                )
        return dataset

    def _load_mono_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given monolingual dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.mono_data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        languages = sorted(
            name for name in self.languages
            if os.path.isdir(os.path.join(data_path, name))
        )
        assert len(languages) == len(self.languages)
        logger.info("Training on {0} languages: {1}".format(len(languages), languages))
        logger.info("Language to id mapping: {0}".format({
                lang: id for id, lang in enumerate(languages)
            })
        )

        lang_datasets = []
        for lang_id, language in enumerate(languages):
            split_path = os.path.join(data_path, language, split)
            lang_dataset = self._load_xlm_dataset(split, split_path, lang1=language, combine=combine, **kwargs)
            lang_datasets.append(lang_dataset)

        dataset = self._maybe_resample_datasets(split, lang_datasets, lang_datasets_meta=languages,
                                                sampling_alpha=self.args.multilang_mono_sampling_alpha,
                                                epoch=epoch, combine=combine, **kwargs)

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))

        assert split not in self.datasets
        self.datasets[split] = SortDataset(
            dataset,
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        # TODO (Leo): we need to modify this for our task
        src_dataset = PadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode='eos',
            ),
            pad_idx=self.source_dictionary.pad(),
            left_pad=False,
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': src_dataset,
                    'src_lengths': NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=1,
    ):
        # Recreate epoch iterator every epoch cause the underlying
        # datasets are dynamic due to sampling.
        self.dataset_to_epoch_iter = {}
        epoch_iter = super().get_batch_iterator(
            dataset, max_tokens, max_sentences, max_positions,
            ignore_invalid_inputs, required_batch_size_multiple,
            seed, num_shards, shard_id, num_workers, epoch,
        )
        self.dataset_to_epoch_iter = {}
        return epoch_iter

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
