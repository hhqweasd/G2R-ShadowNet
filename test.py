import argparse
import os
from os.path import exists, join as join_paths
import torchvision.transforms as transforms
import torch
import numpy as np
from skimage import io, color
from skimage.transform import resize

from model import Generator_S2F,Generator_F2S
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]="0,6,5,1,7,4,2,3"

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_1', type=str, default='ckpt/netG_1.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_2', type=str, default='ckpt/netG_2.pth', help='A2B generator checkpoint file')
opt = parser.parse_args()

## ISTD
opt.dataroot_A = '/home/liuzhihao/dataset/ISTD/test/test_A'
opt.im_suf_A = '.png'
# opt.dataroot_B = '/home/liuzhihao/dataset/ISTD/test/test_B'
opt.dataroot_B = '/home/liuzhihao/BDRAR/test_A_mask_istd_6/'
opt.im_suf_B = '.png'
if torch.cuda.is_available():
    opt.cuda = True
    device = torch.device('cuda:0')
print(opt)


test_448480=0
test_480=1


netG_1 = Generator_S2F()
netG_2 = Generator_F2S()

if opt.cuda:
    netG_1.to(device)
    netG_2.to(device)

gt_list = [os.path.splitext(f)[0] for f in os.listdir(opt.dataroot_A) if f.endswith(opt.im_suf_A)]

for ee in range(100,90,-1):
    g1ckpt='ckpt/netG_1.pth'
    g2ckpt='ckpt/netG_2.pth'
    # g1ckpt='ckpt/netG_1_%s.pth'%(ee)
    # g2ckpt='ckpt/netG_2_%s.pth'%(ee)

    netG_1.load_state_dict(torch.load(g1ckpt))
    netG_1.eval()
    netG_2.load_state_dict(torch.load(g2ckpt))
    netG_2.eval()
    
    savepath='ckpt/B_%s_mask6'%(ee)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    for idx, img_name in enumerate(gt_list):
        # Set model input
        with torch.no_grad():
        
            if test_448480:
                rgbimage=io.imread(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A))
                labimage = color.rgb2lab(io.imread(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A)))   
                
                labimage448=resize(labimage,(448,448,3))
                labimage448[:,:,0]=np.asarray(labimage448[:,:,0])/50.0-1.0
                labimage448[:,:,1:]=2.0*(np.asarray(labimage448[:,:,1:])+128.0)/255.0-1.0
                labimage448=torch.from_numpy(labimage448).float()
                labimage448=labimage448.view(448,448,3)
                labimage448=labimage448.transpose(0, 1).transpose(0, 2).contiguous()
                labimage448=labimage448.unsqueeze(0).to(device)
                
                labimage480=resize(labimage,(480,640,3))
                labimage480[:,:,0]=np.asarray(labimage480[:,:,0])/50.0-1.0
                labimage480[:,:,1:]=2.0*(np.asarray(labimage480[:,:,1:])+128.0)/255.0-1.0
                labimage480=torch.from_numpy(labimage480).float()
                labimage480=labimage480.view(480,640,3)
                labimage480=labimage480.transpose(0, 1).transpose(0, 2).contiguous()
                labimage480=labimage480.unsqueeze(0).to(device)
                
                
                mask=io.imread(os.path.join(opt.dataroot_B, img_name + opt.im_suf_B))
                mask448=resize(mask,(448,448))
                mask448=torch.from_numpy(mask448).float()
                mask448=mask448.view(448,448,1)
                mask448=mask448.transpose(0, 1).transpose(0, 2).contiguous()
                mask448=mask448.unsqueeze(0).to(device)
                # mask0-1
                zero = torch.zeros_like(mask448)
                one = torch.ones_like(mask448)
                mask448=torch.where(mask448 > 0.5, one, zero)
                    
                mask480=resize(mask,(480,640))
                mask480=torch.from_numpy(mask480).float()
                mask480=mask480.view(480,640,1)
                mask480=mask480.transpose(0, 1).transpose(0, 2).contiguous()
                mask480=mask480.unsqueeze(0).to(device)
                # mask0-1
                zero = torch.zeros_like(mask480)
                one = torch.ones_like(mask480)
                mask480=torch.where(mask480 > 0.5, one, zero)
                
                real_s448=labimage448.clone()
                real_s448[:,0]=(real_s448[:,0]+1.0)*mask448-1.0
                real_s448[:,1:]=real_s448[:,1:]*mask448
                
                
                real_s480=labimage480.clone()
                real_s480[:,0]=(real_s480[:,0]+1.0)*mask480-1.0
                real_s480[:,1:]=real_s480[:,1:]*mask480
                
                real_ns448=labimage448.clone()
                real_ns448[:,0]=(real_ns448[:,0]+1.0)*(mask448-1.0)*(-1.0)-1.0
                real_ns448[:,1:]=real_ns448[:,1:]*(mask448-1.0)*(-1.0)
                
                real_ns480=labimage480.clone()
                real_ns480[:,0]=(real_ns480[:,0]+1.0)*(mask480-1.0)*(-1.0)-1.0
                real_ns480[:,1:]=real_ns480[:,1:]*(mask480-1.0)*(-1.0)
                
                temp_B448 = netG_1(real_s448)
                temp_B448 = netG_2(temp_B448+real_ns448,mask448*2.0-1.0)
                
                temp_B480 = netG_1(real_s480)
                temp_B480 = netG_2(temp_B480+real_ns480,mask480*2.0-1.0)
                
                fake_B448 = temp_B448.data
                fake_B448[:,1:]=255.0*(fake_B448[:,1:]+1.0)/2.0-128.0
                fake_B448=fake_B448.data.squeeze(0).cpu()
                fake_B448=fake_B448.transpose(0, 2).transpose(0, 1).contiguous().numpy()
                fake_B448=resize(fake_B448,(480,640,3))
                
                fake_B480 = temp_B480.data
                fake_B480[:,0]=50.0*(fake_B480[:,0]+1.0)
                fake_B480=fake_B480.data.squeeze(0).cpu()
                fake_B480=fake_B480.transpose(0, 2).transpose(0, 1).contiguous().numpy()
                fake_B480=resize(fake_B480,(480,640,3))
                
                fake_B=fake_B480
                fake_B[:,:,1:]=fake_B448[:,:,1:]
                fake_B=color.lab2rgb(fake_B)
                
                mask[mask>0.5]=1
                mask[mask<=0.5]=0
                mask = np.expand_dims(mask, axis=2)
                mask = np.concatenate((mask, mask, mask), axis=-1)
                outputimage=fake_B*mask+rgbimage*(mask-1.0)*(-1.0)/255.0
                save_result = join_paths(savepath+'/%s'% (img_name + opt.im_suf_A))
                io.imsave(save_result,outputimage)
            
            if test_480:
                rgbimage=io.imread(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A))
                labimage = color.rgb2lab(io.imread(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A)))   
                labimage480=resize(labimage,(480,640,3))
                labimage480[:,:,0]=np.asarray(labimage480[:,:,0])/50.0-1.0
                labimage480[:,:,1:]=2.0*(np.asarray(labimage480[:,:,1:])+128.0)/255.0-1.0
                labimage480=torch.from_numpy(labimage480).float()
                labimage480=labimage480.view(480,640,3)
                labimage480=labimage480.transpose(0, 1).transpose(0, 2).contiguous()
                labimage480=labimage480.unsqueeze(0).to(device)
                
                
                mask=io.imread(os.path.join(opt.dataroot_B, img_name + opt.im_suf_B))
                
                mask480=resize(mask,(480,640))
                mask480=torch.from_numpy(mask480).float()
                mask480=mask480.view(480,640,1)
                mask480=mask480.transpose(0, 1).transpose(0, 2).contiguous()
                mask480=mask480.unsqueeze(0).to(device)
                zero = torch.zeros_like(mask480)
                one = torch.ones_like(mask480)
                mask480=torch.where(mask480 > 0.5, one, zero)
                
                real_s480=labimage480.clone()
                real_s480[:,0]=(real_s480[:,0]+1.0)*mask480-1.0
                real_s480[:,1:]=real_s480[:,1:]*mask480

                real_ns480=labimage480.clone()
                real_ns480[:,0]=(real_ns480[:,0]+1.0)*(mask480-1.0)*(-1.0)-1.0
                real_ns480[:,1:]=real_ns480[:,1:]*(mask480-1.0)*(-1.0)
                
                temp_B480 = netG_1(real_s480)
                temp_B480 = netG_2(temp_B480+real_ns480,mask480*2.0-1.0)
        
                fake_B480 = temp_B480.data
                fake_B480[:,0]=50.0*(fake_B480[:,0]+1.0)
                fake_B480[:,1:]=255.0*(fake_B480[:,1:]+1.0)/2.0-128.0
                fake_B480=fake_B480.data.squeeze(0).cpu()
                fake_B480=fake_B480.transpose(0, 2).transpose(0, 1).contiguous().numpy()
                fake_B480=resize(fake_B480,(480,640,3))
                fake_B480=color.lab2rgb(fake_B480)
                
                #replace
                mask[mask>0.5]=1
                mask[mask<=0.5]=0
                mask = np.expand_dims(mask, axis=2)
                mask = np.concatenate((mask, mask, mask), axis=-1)
                outputimage=fake_B480*mask+rgbimage*(mask-1.0)*(-1.0)/255.0
                save_result = join_paths(savepath+'/%s'% (img_name + opt.im_suf_A))
                io.imsave(save_result,outputimage)
            
            print('Generated images %04d of %04d' % (idx+1, len(gt_list)))
    exit(0)