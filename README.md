## Different deep learning models with pytorch

1. Linear Regression
2. Logistic Regression
3. Fully Connected
4. ConvNet
5. RNN

## Writing a Custom Dataloader

~~~~
import torch
from torch.utils import data

class CustomDataloader(data.Dataset):
    def __init__(self, ids, labels):
        self.ids = ids
        self.labels = labels

        # Preprocess the data here
        self.train_data = []
        self.train_labels = []

    def __getitem__(self, index:
        return self.ids[index], self.labels[index]

    def __len__(self):
        return len(len(self.ids))
~~~~

For returning data in batches and shuffling, you can use the torch.utils.data.DataLoader.

~~~~
from torch.utils.data import DataLoader

custom_data = CustomDataLoader()
data_loader = DataLoader(custom_data,batch_size=4, shuffle=True, num_workers=4)
~~~~

Now simply enumerate over ```data_loader```