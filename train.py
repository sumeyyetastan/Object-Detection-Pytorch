import os 
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io 

class Fruit_dataset(Dataset):
    def _init_(self,csv_file,root_dir,transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def _len_(self):
        return len(self.annotations) #10

    def _getitem_(self,index):
        img_path = os.path.join(self.root_dir ,self.annotations.iloc[index,0])
        image =io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index,0]))

        if self.transform:
            image = self.transform(image)

        return(image, y_label)       