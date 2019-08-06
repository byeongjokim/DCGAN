from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class Celeba_Dataset(Dataset):
    #https://wingnim.tistory.com/33
    def __init__(self, imgfolder, transforms=None):
        self.imgfolder = imgfolder
        filenames = os.listdir(self.imgfolder)
        self.imglist = [os.path.join(self.imgfolder, filename) for filename in filenames]
        self.transforms = transforms

    def __getitem__(self, index):
        img_name = self.imglist[index]
        image = Image.open(img_name)

        if self.transforms:
            image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.imglist)