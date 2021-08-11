import json
import os

import torch
import argparse

from pytext import workflow
from pytext.config.serialize import pytext_config_from_json
from pytext.models.roberta import RoBERTa
from pytext.task.serialize import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True,
                        help="path to directory to load pretrained model for finetuning")
    parser.add_argument("--train-filepath", type=str,
                        help="path to train file")
    parser.add_argument("--eval-filepath", type=str,
                        help="path to dev file")
    parser.add_argument("--test-filepath", type=str,
                        help="path to test file")
    parser.add_argument("--model-path", type=str, default=None,
                        help="path to pretrained model file")
    parser.add_argument("--vocab-filepath", type=str, default=None,
                        help="path to vocabulary file")
    parser.add_argument("--spm-model-path", type=str,  default=None,
                        help="path to sentencepiece model file")
    parser.add_argument("--pytext-config-path", type=str, default="xlmr_xnli_finetune_pytext_config.json",
                        help="path to test file")
    args = parser.parse_args()

    config = pytext_config_from_json(json.load(open(args.pytext_config_path, "r")))
    config.task.data.source.train_filename = args.train_filepath
    config.task.data.source.test_filename = args.test_filepath
    config.task.data.source.eval_filename = args.eval_filepath

    config.task.model.inputs.tokens.tokenizer.sp_model_path = args.spm_model_path or os.path.join(
        args.model_dir, "sentencepiece.bpe.model"
    )
    config.task.model.inputs.tokens.vocab_file = args.vocab_filepath or os.path.join(args.model_dir, "dict.txt")
    config.task.model.encoder.model_path = args.model_path or os.path.join(args.model_dir, "model.pt")

    print()
    # trained_model, best_metric = workflow.train_model(config)


if __name__ == "__main__":
    main()