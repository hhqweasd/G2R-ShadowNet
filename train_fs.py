from __future__ import print_function
import os
import datetime
import argparse
import itertools
import torchvision
from torch.utils.data import DataLoader
import torch
from utils import LambdaLR
from utils import weights_init_normal
from model import Generator_S2F,Generator_F2S
from datasets_fs import ImageDataset
import numpy as np
from skimage import io,color
from skimage.transform import resize

import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]="3,7,1,2,0,5,6,4"

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--iter_loss', type=int, default=100, help='average loss for n iterations')
opt = parser.parse_args()


# ISTD
opt.dataroot = '/home/liuzhihao/dataset/ISTD'

if not os.path.exists('ckpt_fs'):
    os.mkdir('ckpt_fs')
opt.log_path = os.path.join('ckpt_fs', str(datetime.datetime.now()) + '.txt')


print(opt)

###### Definition of variables ######
# Networks
netG_1 = Generator_S2F()  # shadow to shadow_free
netG_2 = Generator_F2S()  # shadow to shadow_free

netG_1.cuda()
netG_2.cuda()

netG_1.apply(weights_init_normal)
netG_2.apply(weights_init_normal)


# Lossess
criterion_GAN = torch.nn.MSELoss()  # lsgan
# criterion_GAN = torch.nn.BCEWithLogitsLoss() #vanilla
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_1.parameters(), netG_2.parameters()),lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Dataset loader
dataloader = DataLoader(ImageDataset(opt.dataroot, unaligned=True),batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

curr_iter = 0
G_losses_temp = 0
G_losses = []

open(opt.log_path, 'w').write(str(opt) + '\n\n')

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, (sgt,mask,mask50,sr,srgt,nsr) in enumerate(dataloader):
        # Set model input
        sgt=sgt.cuda()
        mask=mask.cuda()
        mask50=mask50.cuda()
        sr=sr.cuda()
        srgt=srgt.cuda()
        nsr=nsr.cuda()
        
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        fake_nsr=netG_1(sr)
        loss_cycle=criterion_cycle(fake_nsr,srgt)
        
        output=netG_2(fake_nsr+nsr,mask*2.0-1.0)
        loss_sr=criterion_identity(output,sgt)
        
        loss_shadow=criterion_cycle(torch.cat(((output[:,0]+1.0)*mask50-1.0,output[:,1:]*mask50),1),torch.cat(((sgt[:,0]+1.0)*mask50-1.0,sgt[:,1:]*mask50),1))
        

        # Total loss
        loss_G=loss_cycle+loss_sr+loss_shadow
        loss_G.backward()

        G_losses_temp += loss_G.item()

        optimizer_G.step()
        ###################################

        curr_iter += 1

        if (i+1) % opt.iter_loss == 0:
            log = 'Epoch: %d, [iter %d], [loss_G %.5f], [loss_1 %.5f], [loss_2 %.5f], [loss_shadow %.5f]' % \
                  (epoch, curr_iter, loss_G,loss_cycle,loss_sr,loss_shadow)
            print(log)
            open(opt.log_path, 'a').write(log + '\n')

            G_losses.append(G_losses_temp / opt.iter_loss)
            G_losses_temp = 0

            avg_log = '[the last %d iters], [loss_G %.5f]'% (opt.iter_loss, G_losses[G_losses.__len__()-1])
            print(avg_log)
            open(opt.log_path, 'a').write(avg_log + '\n')
            

            slabimage480_realsr=output.data
            slabimage480_realsr[:,0]=50.0*(slabimage480_realsr[:,0]+1.0)
            slabimage480_realsr[:,1:]=255.0*(slabimage480_realsr[:,1:]+1.0)/2.0-128.0
            slabimage480_realsr=slabimage480_realsr.data.squeeze(0).cpu()
            slabimage480_realsr=slabimage480_realsr.transpose(0, 2).transpose(0, 1).contiguous().numpy()
            slabimage480_realsr=resize(slabimage480_realsr,(480,640,3))
            outputimagerealsr=color.lab2rgb(slabimage480_realsr)
            io.imsave('./ckpt_fs/fake.png',(outputimagerealsr*255).astype(np.uint8))
            
            
    # Update learning rates
    lr_scheduler_G.step()


    if epoch>90:
        torch.save(netG_1.state_dict(), ('ckpt_fs/netG_1_%d.pth' % (epoch + 1)))
        torch.save(netG_2.state_dict(), ('ckpt_fs/netG_2_%d.pth' % (epoch + 1)))

    print('Epoch:{}'.format(epoch))