import glob
import random
import os

from torch.utils.data import Dataset
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
import random
import numpy as np
import torch
#这个默认是用膨胀核大小为50的
class ImageDataset(Dataset):
    def __init__(self, root, unaligned=False, mode='train'):
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/train_A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/train_D' % mode) + '/*.*'))
        self.files_C = sorted(glob.glob(os.path.join(root, '%s/train_B' % mode) + '/*.*'))
        self.files_D = sorted(glob.glob(os.path.join(root, '%s/train_C' % mode) + '/*.*'))
        self.files_E = sorted(glob.glob(os.path.join(root, '%s/train_E' % mode) + '/*.*'))
        self.files_F = sorted(glob.glob(os.path.join(root, '%s/train_F' % mode) + '/*.*'))

    def __getitem__(self, index):
        i = random.randint(0, 48)
        j = random.randint(0, 48)
        k=random.randint(0,100)
        
        item_A=color.rgb2lab(io.imread(self.files_A[index % len(self.files_A)]))
        item_A=resize(item_A,(448,448,3))
        item_A=item_A[i:i+400,j:j+400,:]
        if k>50:
            item_A=np.fliplr(item_A)
        item_A[:,:,0]=np.asarray(item_A[:,:,0])/50.0-1.0
        item_A[:,:,1:]=2.0*(np.asarray(item_A[:,:,1:])+128.0)/255.0-1.0
        item_A=torch.from_numpy(item_A.copy()).float()
        item_A=item_A.view(400,400,3)
        item_A=item_A.transpose(0, 1).transpose(0, 2).contiguous()
        if self.unaligned:
            item_B = color.rgb2lab(io.imread(self.files_B[random.randint(0, len(self.files_B) - 1)]))
            item_B=resize(item_B,(448,448,3))
            item_B=item_B[i:i+400,j:j+400,:]
            if k>50:
                item_B=np.fliplr(item_B)
            item_B[:,:,0]=np.asarray(item_B[:,:,0])/50.0-1.0
            item_B[:,:,1:]=2.0*(np.asarray(item_B[:,:,1:])+128.0)/255.0-1.0
            item_B=torch.from_numpy(item_B.copy()).float()
            item_B=item_B.view(400,400,3)
            item_B=item_B.transpose(0, 1).transpose(0, 2).contiguous()
        else:
            item_B = color.rgb2lab(io.imread(self.files_B[index % len(self.files_B)]))
            item_B=resize(item_B,(448,448,3))
            item_B=item_B[i:i+400,j:j+400,:]
            if k>50:
                item_B=np.fliplr(item_B)
            item_B[:,:,0]=np.asarray(item_B[:,:,0])/50.0-1.0
            item_B[:,:,1:]=2.0*(np.asarray(item_B[:,:,1:])+128.0)/255.0-1.0
            item_B=torch.from_numpy(item_B.copy()).float()
            item_B=item_B.view(400,400,3)
            item_B=item_B.transpose(0, 1).transpose(0, 2).contiguous()
            
        item_C=io.imread(self.files_C[index % len(self.files_C)])
        item_C=resize(item_C[:,:,0],(448,448))
        item_C=item_C[i:i+400,j:j+400]
        if k>50:
            item_C=np.fliplr(item_C)
        #item_C=2.0*np.asarray(item_C)/255.0-1.0
        item_C=torch.from_numpy(item_C.copy()).float()
        item_C=item_C.view(400,400,1)
        item_C=item_C.transpose(0, 1).transpose(0, 2).contiguous()
        
        
        item_D=color.rgb2lab(io.imread(self.files_D[index % len(self.files_D)]))
        item_D=resize(item_D,(448,448,3))
        item_D=item_D[i:i+400,j:j+400,:]
        if k>50:
            item_D=np.fliplr(item_D)
        item_D[:,:,0]=np.asarray(item_D[:,:,0])/50.0-1.0
        item_D[:,:,1:]=2.0*(np.asarray(item_D[:,:,1:])+128.0)/255.0-1.0
        item_D=torch.from_numpy(item_D.copy()).float()
        item_D=item_D.view(400,400,3)
        item_D=item_D.transpose(0, 1).transpose(0, 2).contiguous()
        
        item_E=color.rgb2lab(io.imread(self.files_E[index % len(self.files_E)]))
        item_E=resize(item_E,(448,448,3))
        item_E=item_E[i:i+400,j:j+400,:]
        if k>50:
            item_E=np.fliplr(item_E)
        item_E[:,:,0]=np.asarray(item_E[:,:,0])/50.0-1.0
        item_E[:,:,1:]=2.0*(np.asarray(item_E[:,:,1:])+128.0)/255.0-1.0
        item_E=torch.from_numpy(item_E.copy()).float()
        item_E=item_E.view(400,400,3)
        item_E=item_E.transpose(0, 1).transpose(0, 2).contiguous()
        
        
        item_F=io.imread(self.files_F[index % len(self.files_F)])
        item_F=resize(item_F[:,:,0],(448,448))
        item_F=item_F[i:i+400,j:j+400]
        if k>50:
            item_F=np.fliplr(item_F)
        #item_F=2.0*np.asarray(item_F)/255.0-1.0
        item_F=torch.from_numpy(item_F.copy()).float()
        item_F=item_F.view(400,400,1)
        item_F=item_F.transpose(0, 1).transpose(0, 2).contiguous()
          
        return {'A': item_A, 'B': item_B, 'C': item_C, 'D': item_D, 'E': item_E, 'F': item_F}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B), len(self.files_C), len(self.files_D), len(self.files_E), len(self.files_F))

