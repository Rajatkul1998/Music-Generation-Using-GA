import torch
from torch.utils.data import DataLoader
class Dataset:
    def __init__(self,main_list,batch_size):
        super().__init__()
        self.main_list=main_list
        self.batch_size=batch_size

    def __len__(self):
        return len(self.main_list)


    def convert_to_tensor(self):
        return torch.FloatTensor(self.main_list)    

    def dataloader(self,tensor):
        return DataLoader(tensor,batch_size=self.batch_size,drop_last=True)          
