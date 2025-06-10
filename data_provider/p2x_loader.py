import pickle
from torch.utils.data import Dataset
import torch

class P2XDataset(Dataset):
    """Dataset for p2x EHR classification tasks."""

    def __init__(self, pkl_path: str, task: str):
        """Load dataset from ``pkl_path``.

        Parameters
        ----------
        pkl_path : str
            Path to pickle file containing a list of dictionaries.
        task : {'ihm', 'pheno'}
            Downstream task type.
        """
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        self.task = task
        self.seq_len = self.data[0]['reg_ts'].shape[0]
        self.num_features = 17
        self.num_classes = 24 if task == 'pheno' else 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ts = torch.tensor(item['reg_ts'][:, :self.num_features], dtype=torch.float32)
        text = ' '.join(item['text_data'])
        if self.task == 'pheno':
            label = torch.tensor(item['label'][1:25], dtype=torch.float32)
        else:
            label = torch.tensor(item['label'], dtype=torch.long)
        return ts, text, label
