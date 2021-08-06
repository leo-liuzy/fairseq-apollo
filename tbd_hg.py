import numpy as np
import fairseq
from fairseq import utils
from transformers import AutoModelForMaskedLM
import torch

seed = 42
np.random.seed(seed)
utils.set_torch_seed(seed)

hg_model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
hg_model.eval()
input = torch.load('full_input.pt')
output = hg_model(input['net_input']['src_tokens_mlm'],
                  attention_mask=input['net_input']['src_tokens_mlm'].ne(1),
                  labels=input['net_input']['src_tokens'])
print()