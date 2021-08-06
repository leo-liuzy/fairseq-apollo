import numpy as np
import fairseq
from fairseq import utils, modules
from fairseq.models.roberta.model_xlmr import XLMRXCLModel
import torch


seed = 42
np.random.seed(seed)
utils.set_torch_seed(seed)

model = XLMRXCLModel.from_pretrained("data/xlmr.base").model
model.eval()
input = torch.load('mlm_input.pt')
masked_tokens = input['net_input']['src_tokens'].eq(250001)
logits = model(**input['net_input'], force_positions=True)[0]
masked_tokens_logits = logits[masked_tokens, :]
targets = model.get_targets(input, [masked_tokens_logits])
if masked_tokens is not None:
    targets = targets[masked_tokens]
loss = modules.cross_entropy(
            masked_tokens_logits.view(-1, masked_tokens_logits.size(-1)),
            targets.view(-1),
            reduction='sum',
            ignore_index=1,
        )

print()