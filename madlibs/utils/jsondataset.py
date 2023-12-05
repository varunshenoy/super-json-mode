from torch.utils.data import Dataset

class JSONDataset(Dataset):
    def __init__(self, data):
      self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]