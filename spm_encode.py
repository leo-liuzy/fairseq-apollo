import os
import sys
import argparse
from tqdm import tqdm
import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Pretrained Sentenpiece model file")
args = parser.parse_args()
# output="/Users/leoliu/proj/fairseq-apollo/data/tokenized_XLM_pilot_run_21Langs_debug/monolingual/test.zh-Hans"

# sys.stdout = open(output, "w")
assert os.path.exists(args.model)
sp = spm.SentencePieceProcessor(model_file=args.model)

for line in tqdm(sys.stdin):
    encoded_line = sp.encode(line, out_type=str)
    sys.stdout.write(" ".join(encoded_line))

print("Done")