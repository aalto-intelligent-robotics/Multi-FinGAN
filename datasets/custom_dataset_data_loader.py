import torch.utils.data
from datasets.dataset import DatasetFactory


class CustomDatasetDataLoader:
    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.mode = mode
        self.num_threds = opt.n_threads_train
        self.create_dataset()

    def create_dataset(self):
        self.dataset = DatasetFactory.get_by_name(
            self.opt.dataset_name, self.opt, self.mode)
        if hasattr(self.dataset, 'collate_fn'):
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.opt.batch_size,
                collate_fn=self.dataset.collate_fn,
                shuffle=not self.opt.serial_batches and self.mode == 'train',
                num_workers=int(self.num_threds),
                drop_last=True)
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.opt.batch_size,
                shuffle=not self.opt.serial_batches and self.mode == 'train',
                num_workers=int(self.num_threds),
                drop_last=True)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
