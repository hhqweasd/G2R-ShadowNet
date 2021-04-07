import glob
import random
import os

from torch.utils.data import Dataset
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
import random
import numpy as np
import torch

class ImageDataset(Dataset):
    def __init__(self, root, unaligned=False, mode='train'):
        self.unaligned = unaligned

        self.files_B = sorted(glob.glob(os.path.join(root, '%s/train_C_fixed_official' % mode) + '/*.*'))
        self.files_mask = sorted(glob.glob(os.path.join(root, '%s/train_B' % mode) + '/*.*'))
        self.files_mask50 = sorted(glob.glob(os.path.join(root, '%s/train_mask50' % mode) + '/*.*'))
        self.files_sr = sorted(glob.glob(os.path.join(root, '%s/train_sr' % mode) + '/*.*'))
        self.files_srgt = sorted(glob.glob(os.path.join(root, '%s/train_srgt' % mode) + '/*.*'))
        self.files_nsr = sorted(glob.glob(os.path.join(root, '%s/train_nsr' % mode) + '/*.*'))

    def __getitem__(self, index):
        i = random.randint(0, 48)
        j = random.randint(0, 48)
        k=random.randint(0,100)
        
        item_B=color.rgb2lab(io.imread(self.files_B[index % len(self.files_B)]))
        item_B=resize(item_B,(448,448,3))
        item_B=item_B[i:i+400,j:j+400,:]
        if k>50:
            item_B=np.fliplr(item_B)
        item_B[:,:,0]=np.asarray(item_B[:,:,0])/50.0-1.0
        item_B[:,:,1:]=2.0*(np.asarray(item_B[:,:,1:])+128.0)/255.0-1.0
        item_B=torch.from_numpy(item_B.copy()).float()
        item_B=item_B.view(400,400,3)
        item_B=item_B.transpose(0, 1).transpose(0, 2).contiguous()

        item_mask=io.imread(self.files_mask[index % len(self.files_mask)])
        item_mask=resize(item_mask,(448,448,1))
        item_mask=item_mask[i:i+400,j:j+400,:]
        item_mask[item_mask>0] = 1.0
        if k>50:
            item_mask=np.fliplr(item_mask)
        item_mask=np.asarray(item_mask)
        item_mask=torch.from_numpy(item_mask.copy()).float()
        item_mask=item_mask.view(400,400,1)
        item_mask=item_mask.transpose(0, 1).transpose(0, 2).contiguous()
        
        item_mask50=io.imread(self.files_mask50[index % len(self.files_mask50)])
        item_mask50=resize(item_mask50,(448,448,1))
        item_mask50=item_mask50[i:i+400,j:j+400,:]
        item_mask50[item_mask50>0] = 1.0
        if k>50:
            item_mask50=np.fliplr(item_mask50)
        item_mask50=np.asarray(item_mask50)
        item_mask50=torch.from_numpy(item_mask50.copy()).float()
        item_mask50=item_mask50.view(400,400,1)
        item_mask50=item_mask50.transpose(0, 1).transpose(0, 2).contiguous()
        
        item_sr=color.rgb2lab(io.imread(self.files_sr[index % len(self.files_sr)]))
        item_sr=resize(item_sr,(448,448,3))
        item_sr=item_sr[i:i+400,j:j+400,:]
        if k>50:
            item_sr=np.fliplr(item_sr)
        item_sr[:,:,0]=np.asarray(item_sr[:,:,0])/50.0-1.0
        item_sr[:,:,1:]=2.0*(np.asarray(item_sr[:,:,1:])+128.0)/255.0-1.0
        item_sr=torch.from_numpy(item_sr.copy()).float()
        item_sr=item_sr.view(400,400,3)
        item_sr=item_sr.transpose(0, 1).transpose(0, 2).contiguous()
        
        item_srgt=color.rgb2lab(io.imread(self.files_sr[index % len(self.files_sr)]))
        item_srgt=resize(item_srgt,(448,448,3))
        item_srgt=item_srgt[i:i+400,j:j+400,:]
        if k>50:
            item_srgt=np.fliplr(item_srgt)
        item_srgt[:,:,0]=np.asarray(item_srgt[:,:,0])/50.0-1.0
        item_srgt[:,:,1:]=2.0*(np.asarray(item_srgt[:,:,1:])+128.0)/255.0-1.0
        item_srgt=torch.from_numpy(item_srgt.copy()).float()
        item_srgt=item_srgt.view(400,400,3)
        item_srgt=item_srgt.transpose(0, 1).transpose(0, 2).contiguous()
        
        item_nsr=color.rgb2lab(io.imread(self.files_nsr[index % len(self.files_nsr)]))
        item_nsr=resize(item_nsr,(448,448,3))
        item_nsr=item_nsr[i:i+400,j:j+400,:]
        if k>50:
            item_nsr=np.fliplr(item_nsr)
        item_nsr[:,:,0]=np.asarray(item_nsr[:,:,0])/50.0-1.0
        item_nsr[:,:,1:]=2.0*(np.asarray(item_nsr[:,:,1:])+128.0)/255.0-1.0
        item_nsr=torch.from_numpy(item_nsr.copy()).float()
        item_nsr=item_nsr.view(400,400,3)
        item_nsr=item_nsr.transpose(0, 1).transpose(0, 2).contiguous()



        return item_B,item_mask,item_mask50,item_sr,item_srgt,item_nsr

    def __len__(self):
        return max(len(self.files_B),len(self.files_mask))

