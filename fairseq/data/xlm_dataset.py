from . import FairseqDataset


class XlmDataset(FairseqDataset):
    """
    This dataset mimic the implementation of XLM with memory efficiency
    In TLM, one might want sentences ordered in
    <en, fr> and <fr, en>
    However, doubling the parallel corpus seems wasteful, we will achieve
    effectively the same thing with the help of indexing.
    Note: this dataset needs to be used by calling twice; for example, in en-fr
        For source language, one needs XlmDataset(en, fr)
        For target language, one needs XlmDataset(fr, en)
        We highlight that the underlying dataset that actually save the
         data is not doubled, only references are doubled

    """
    def __init__(self, lang1_dataset, lang2_dataset=None):
        super(XlmDataset, self).__init__()
        assert len(lang1_dataset) > 0, "datasets should not be an empty iterable"
        self.tlm = lang2_dataset is not None
        if lang2_dataset:
            assert len(lang2_dataset) > 0, "datasets should not be an empty iterable"
            assert len(lang1_dataset) == len(lang2_dataset)
        self.datasets = [lang1_dataset, lang2_dataset] if lang2_dataset else [lang1_dataset]
        self._size_of_one_dataset = len(lang1_dataset)
        self.size = self._size_of_one_dataset * len(self.datasets)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.tlm:
            return self.datasets[0][idx]
        else:
            dataset_idx = idx // self._size_of_one_dataset
            sample_idx = idx % self._size_of_one_dataset
