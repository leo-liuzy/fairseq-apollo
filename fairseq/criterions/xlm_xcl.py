# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist

from collections import defaultdict
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from ipdb import set_trace as bp


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temperature


@register_criterion('xlm_xcl')
class XlmXclLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, task, tpu):
        super().__init__(task)
        self.tpu = tpu
        assert hasattr(task, "objective_use_mode")
        self.objective_use_mode = task.objective_use_mode
        if self.objective_use_mode == 'alter':
            self.use_mono = True  # internal counter of whether we use mono or para objective
        # metric used by contrastive learning
        self.mask_idx = task.mask_idx
        self.mcl_similarity_metric = Similarity(self.task.args.temp_mcl)
        self.tcl_similarity_metric = Similarity(self.task.args.temp_tcl)
        # loss coefficient
        self.mlm_coeff = self.task.args.mlm_coeff
        self.tlm_coeff = self.task.args.tlm_coeff
        self.mcl_coeff = self.task.args.mcl_coeff
        self.tcl_coeff = self.task.args.tcl_coeff

        self.use_mlm = self.task.args.use_mlm
        self.use_tlm = self.task.args.use_tlm
        self.use_mcl = self.task.args.use_mcl
        self.use_tcl = self.task.args.use_tcl
        assert any([self.use_mlm, self.use_tlm, self.use_mcl, self.use_tcl])

    def _mask_helper(self, mask):
        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if self.tpu:
            mask = None  # always project all tokens on TPU
        elif mask.device == torch.device('cpu'):
            if not mask.any():
                mask = None
        else:
            mask = torch.where(
                mask.any(),
                mask,
                mask.new([True]),
            )
        return mask

    def _get_attn_mask(self, token_ids):
        mask = token_ids.ne(self.padding_idx)
        mask = self._mask_helper(mask)
        return mask

    def _get_masked_tokens(self, token_ids):
        mask = token_ids.eq(self.mask_idx)
        mask = self._mask_helper(mask)
        return mask

    def _xlm_forward(self, model, sample, log_prefix, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        model.train()
        masked_tokens = self._get_masked_tokens(sample['net_input']['src_tokens'])
        sample_size = masked_tokens.int().sum()
        logits = model(**sample['net_input'])[0]
        masked_tokens_logits = logits[masked_tokens, :]
        targets = model.get_targets(sample, [logits])
        if masked_tokens is not None:
            targets = targets[masked_tokens]

        loss = modules.cross_entropy(
            masked_tokens_logits.view(-1, masked_tokens_logits.size(-1)),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        logging_output = {
            f'{log_prefix}_loss': loss if self.tpu else loss.data,
            f'{log_prefix}_ntokens': sample['ntokens'],
            f'{log_prefix}_nsentences': sample['nsentences'],
            f'{log_prefix}_sample_size': sample_size,
        }

        return loss, sample_size, logging_output

    def mlm_forward(self, model, sample, reduce=True):
        """Make a forward pass for MLM"""
        mlm_sample = {
            'net_input': {
                "src_tokens": sample["net_input"]['src_tokens_mlm'],
                "src_positions": sample["net_input"]['src_positions'],
                "src_lengths": sample["net_input"]['src_lengths'],
            },
            'target': sample['target']['src_mlm'],
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences']
        }

        return self._xlm_forward(model, mlm_sample, 'mlm', reduce)

    def tlm_forward(self, model, sample, reduce=True):
        tlm_sample = {
            'net_input': {
                "src_tokens": torch.cat([sample["net_input"]['src_tokens_mlm'],
                                         sample["net_input"]['tgt_tokens_mlm']], dim=-1),
                "src_positions": torch.cat([sample["net_input"]['src_positions'],
                                            sample["net_input"]['tgt_positions']], dim=-1),
                "src_lengths": sample["net_input"]['src_lengths'] + sample["net_input"]['tgt_lengths'],
            },
            'target': torch.cat([sample["target"]['src_mlm'],
                                 sample["target"]['tgt_mlm']], dim=-1),
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences']
        }

        return self._xlm_forward(model, tlm_sample, 'tlm', reduce)

    def _calculate_cl(self, sentence_rep_x, sentence_rep_z, sim_metric):
        """
        Calculate the contrastive loss, given
            sentence_rep_x --- the regular sentence representations from the encoder
        and sentence_rep_z --- candidate representations used to make positive or negative pairs

        MCL -> sentence_rep_z is obtained from calculating the same input with a different dropout mask
        TCL -> sentence_rep_z is obtained from calculating the sentence representations of parallel sentences
        """
        rank = 0
        batch_size, _ = sentence_rep_x.shape
        # bp()
        if dist.is_initialized():
            # Dummy vectors for allgather
            x_list = [torch.zeros_like(sentence_rep_x) for _ in range(dist.get_world_size())]
            z_list = [torch.zeros_like(sentence_rep_z) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=x_list, tensor=sentence_rep_x.contiguous())
            dist.all_gather(tensor_list=z_list, tensor=sentence_rep_z.contiguous())
            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            rank = dist.get_rank()
            x_list[rank] = sentence_rep_x
            z_list[rank] = sentence_rep_z
            # Get full batch embeddings: (batch_size x num_workers, hidden)
            # get all
            sentence_rep_x = torch.cat(x_list, 0)
                sentence_rep_z = torch.cat(z_list, 0)
            bp()
            # print(sentence_rep_x.shape)
        # (batch_size*num_workers x batch_size*num_workers)
        cos_sim = sim_metric(sentence_rep_x.unsqueeze(1), sentence_rep_z.unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long()
        # TODO(Leo): find the right chunk for each worker on cos_sim and labels
        cos_sim = cos_sim[rank*batch_size:(rank + 1)*batch_size]
        labels = labels[rank*batch_size:(rank + 1)*batch_size]
        # print(f"rank: {rank}")
        # print(f"Cur lower: {rank*batch_size}, Cur upper: {(rank + 1)*batch_size}")
        return cos_sim, labels

    def _get_cl_logs(self, log_prefix, loss, sample, sample_size, cos_sim):
        similarities_with_positive = cos_sim.diag()
        assert len(similarities_with_positive.shape) == 1
        mask = torch.ones_like(cos_sim).fill_diagonal_(0)
        similarities_with_negative = torch.masked_select(cos_sim, mask.bool()).view(sample_size, -1)

        logging_output = {
            f'{log_prefix}_loss': loss if self.tpu else loss.data,
            f'{log_prefix}_ntokens': sample['ntokens'],
            f'{log_prefix}_nsentences': sample['nsentences'],
            f'{log_prefix}_sample_size': sample_size,
            f'{log_prefix}_sim_positive': similarities_with_positive,
            f'{log_prefix}_sim_negative_mean': torch.mean(similarities_with_negative, dim=-1),
            f'{log_prefix}_sim_negative_std': torch.std(similarities_with_negative, dim=-1)
        }
        return logging_output
    
    def mcl_forward(self, model, sample, reduce=True):
        assert hasattr(model, "encoder")
        assert hasattr(model.encoder, "extract_features"), "Require model to have feature extractor"
        assert hasattr(model, "pooler"), "Require model to have pooler for sentence representation"
        encoder = model.encoder
        attn_mask = self._get_attn_mask(sample['net_input']["src_tokens"])
        sample_size = len(sample['net_input']["src_tokens"])

        # x_extra['inner_states'][0] -> embedding layer
        x, x_extra = encoder.extract_features(src_tokens=sample['net_input']['src_tokens'],
                                              src_positions=sample['net_input']['src_positions'],
                                              force_positions=True,
                                              return_all_hiddens=True)
        sentence_rep_x = model.pooler(attn_mask, x_extra['inner_states'])
        # z_extra['inner_states'][0] -> embedding layer
        z, z_extra = encoder.extract_features(src_tokens=sample['net_input']['src_tokens'],
                                              src_positions=sample['net_input']['src_positions'],
                                              force_positions=True,
                                              return_all_hiddens=True)
        sentence_rep_z = model.pooler(attn_mask, z_extra['inner_states'])
        cos_sim, labels = self._calculate_cl(sentence_rep_x, sentence_rep_z, self.mcl_similarity_metric)
        labels = labels.to(model.encoder.sentence_encoder.embed_tokens.weight.device)
        # print(f"cos_sim shape: {cos_sim.shape}")
        # print(f"labels: {labels}")
        loss_fct = nn.CrossEntropyLoss()
        mcl_loss = loss_fct(cos_sim, labels)
        logging_output = self._get_cl_logs(log_prefix="mcl",
                                           loss=mcl_loss,
                                           sample=sample,
                                           sample_size=sample_size,
                                           cos_sim=cos_sim)

        return mcl_loss, sample_size, logging_output

    def tcl_forward(self, model, sample, reduce=True):
        assert hasattr(model, "encoder")
        assert hasattr(model.encoder, "extract_features"), "Require model to have feature extractor"
        assert hasattr(model, "pooler"), "Require model to have pooler for sentence representation"
        encoder = model.encoder
        x_attn_mask = self._get_attn_mask(sample['net_input']["src_tokens"])
        sample_size = len(sample['net_input']["src_tokens"])
        # x_extra['inner_states'][0] -> embedding layer
        x, x_extra = encoder.extract_features(src_tokens=sample['net_input']['src_tokens'],
                                              src_positions=sample['net_input']['src_positions'],
                                              force_positions=True,
                                              return_all_hiddens=True)
        sentence_rep_x = model.pooler(x_attn_mask, x_extra['inner_states'])
        z_attn_mask = self._get_attn_mask(sample['net_input']["tgt_tokens"])
        z, z_extra = encoder.extract_features(src_tokens=sample['net_input']['tgt_tokens'],
                                              src_positions=sample['net_input']['tgt_positions'],
                                              force_positions=True,
                                              return_all_hiddens=True)
        sentence_rep_z = model.pooler(z_attn_mask, z_extra['inner_states'])
        cos_sim, labels = self._calculate_cl(sentence_rep_x, sentence_rep_z, self.tcl_similarity_metric)
        labels = labels.to(model.encoder.sentence_encoder.embed_tokens.weight.device)
        loss_fct = nn.CrossEntropyLoss()
        tcl_loss = loss_fct(cos_sim, labels)
        logging_output = self._get_cl_logs(log_prefix="tcl",
                                           loss=tcl_loss,
                                           sample=sample,
                                           sample_size=sample_size,
                                           cos_sim=cos_sim)

        return tcl_loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss = 0
        sample_size = 0
        logging_output = defaultdict(int)
        if len(sample['target']) == 1:
            # monolingual
            if self.use_mlm:
                mlm_loss, mlm_sample_size, mlm_logging_output = self.mlm_forward(model, sample, reduce)
                loss += self.mlm_coeff * mlm_loss
                sample_size += mlm_sample_size
                logging_output.update(mlm_logging_output)
            # mcl
            if self.use_mcl:
                mcl_loss, mcl_sample_size, mcl_logging_output = self.mcl_forward(model, sample, reduce)
                loss += self.mcl_coeff * mcl_loss
                sample_size += mcl_sample_size
                logging_output.update(mcl_logging_output)
            logging_output.update({"mono": True})  # we use this as a indicator for reduce_metrics()

        elif len(sample['target']) == 2:
            # tlm
            if self.use_tlm:
                tlm_loss, tlm_sample_size, tlm_logging_output = self.tlm_forward(model, sample, reduce)
                loss += self.tlm_coeff * tlm_loss
                sample_size += tlm_sample_size
                logging_output.update(tlm_logging_output)

            # tcl
            if self.use_tcl:
                tcl_loss, tcl_sample_size, tcl_logging_output = self.tcl_forward(model, sample, reduce)
                loss += self.tcl_coeff * tcl_loss
                sample_size += tcl_sample_size
                logging_output.update(tcl_logging_output)
            logging_output.update({"mono": False})
        else:
            raise Exception("Invalid data format.")
        # TODO (Leo): understand how 'sample_size' is used
        logging_output['ntokens'] = sum(v for k, v in logging_output.items() if 'ntokens' in k)
        logging_output['nsentences'] = np.mean([v for k, v in logging_output.items() if 'nsentences' in k])
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        assert len(logging_outputs) > 0
        loss = 0
        sample_size = 0
        if logging_outputs[0]["mono"]:
            # in monolingual mode
            if "mlm_loss" in logging_outputs[0]:
                mlm_loss_sum = sum(log.get('mlm_loss', 0) for log in logging_outputs)
                mlm_sample_size = sum(log.get('mlm_sample_size', 0) for log in logging_outputs)
                mean_mlm_loss = mlm_loss_sum / mlm_sample_size
                metrics.log_scalar('mlm_loss', mean_mlm_loss / math.log(2), mlm_sample_size, round=3)
                metrics.log_derived('mlm_ppl', lambda meters: utils.get_perplexity(meters['mlm_loss'].avg))
                loss += mean_mlm_loss / math.log(2)
                sample_size += mlm_sample_size

            if "mcl_loss" in logging_outputs[0]:
                mcl_loss_sum = sum(log.get('mcl_loss', 0) for log in logging_outputs)
                mcl_sample_size = sum(log.get('mcl_sample_size', 0) for log in logging_outputs)
                mcl_sim_positive_sum = sum(sum(log.get('mcl_sim_positive', 0) for log in logging_outputs))
                mcl_sim_negative_sum_mean = sum(sum(log.get('mcl_sim_negative_mean', 0) for log in logging_outputs))
                mcl_sim_negative_sum_std = sum(sum(log.get('mcl_sim_negative_std', 0) for log in logging_outputs))

                mean_mcl_loss = mcl_loss_sum / mcl_sample_size
                metrics.log_scalar('mcl_loss', mean_mcl_loss, mcl_sample_size, round=3)
                metrics.log_scalar('mcl_sim_positive_mean', mcl_sim_positive_sum / mcl_sample_size, mcl_sample_size,
                                   round=3)
                metrics.log_scalar('mcl_sim_negative_mean_mean', mcl_sim_negative_sum_mean / mcl_sample_size,
                                   mcl_sample_size, round=3)
                metrics.log_scalar('mcl_sim_negative_mean_std', mcl_sim_negative_sum_std / mcl_sample_size,
                                   mcl_sample_size, round=3)

                loss += mean_mcl_loss
                loss /= 2 if "mcl_loss" in logging_outputs[0] and "mlm_loss" in logging_outputs[0] else 1
                sample_size += mcl_sample_size

            metrics.log_scalar('loss', loss, sample_size, round=3)

        else:
            # in bilingual mode
            if "tlm_loss" in logging_outputs[0]:
                tlm_loss_sum = sum(log.get('tlm_loss', 0) for log in logging_outputs)
                tlm_sample_size = sum(log.get('tlm_sample_size', 0) for log in logging_outputs)
                mean_tlm_loss = tlm_loss_sum / tlm_sample_size
                metrics.log_scalar('tlm_loss', mean_tlm_loss / math.log(2), tlm_sample_size, round=3)
                metrics.log_derived('tlm_ppl', lambda meters: utils.get_perplexity(meters['tlm_loss'].avg))
                loss += mean_tlm_loss / math.log(2)
                sample_size += tlm_sample_size

            if "tcl_loss" in logging_outputs[0]:
                tcl_loss_sum = sum(log.get('tcl_loss', 0) for log in logging_outputs)
                tcl_sample_size = sum(log.get('tcl_sample_size', 0) for log in logging_outputs)
                tcl_sim_positive_sum = sum(sum(log.get('tcl_sim_positive', 0) for log in logging_outputs))
                tcl_sim_negative_sum_mean = sum(sum(log.get('tcl_sim_negative_mean', 0) for log in logging_outputs))
                tcl_sim_negative_sum_std = sum(sum(log.get('tcl_sim_negative_std', 0) for log in logging_outputs))

                mean_tcl_loss = tcl_loss_sum / tcl_sample_size
                metrics.log_scalar('tcl_loss', mean_tcl_loss, tcl_sample_size, round=3)
                metrics.log_scalar('mcl_sim_positive_mean', tcl_sim_positive_sum / tcl_sample_size, tcl_sample_size,
                                   round=3)
                metrics.log_scalar('mcl_sim_negative_mean_mean', tcl_sim_negative_sum_mean / tcl_sample_size,
                                   tcl_sample_size, round=3)
                metrics.log_scalar('mcl_sim_negative_mean_std', tcl_sim_negative_sum_std / tcl_sample_size,
                                   tcl_sample_size, round=3)
                loss += mean_tcl_loss
                loss /= 2 if "tcl_loss" in logging_outputs[0] and "tlm_loss" in logging_outputs[0] else 1
                sample_size += tcl_sample_size
            metrics.log_scalar('loss', loss, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
