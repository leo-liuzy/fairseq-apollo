from fairseq.models.roberta import XLMRModel
xlmr = XLMRModel.from_pretrained('xlmr.base', checkpoint_file='model.pt')
print()