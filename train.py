from __future__ import print_function
import os
import datetime
import argparse
import itertools
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal
from model import Generator_S2F,Generator_F2S,Discriminator
from datasets import ImageDataset

os.environ["CUDA_VISIBLE_DEVICES"]="7,3,1,2,0,5,6,4"
torch.manual_seed(628)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=400, help='size of the data crop (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--snapshot_epochs', type=int, default=5, help='number of epochs of training')
parser.add_argument('--iter_loss', type=int, default=100, help='average loss for n iterations')
opt = parser.parse_args()


# ISTD
opt.dataroot = '/home/liuzhihao/dataset/PAISTD8'

if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
opt.log_path = os.path.join('ckpt', str(datetime.datetime.now()) + '.txt')

if torch.cuda.is_available():
    opt.cuda = True

print(opt)

###### Definition of variables ######
# Networks
netG_A2B = Generator_S2F()  # shadow to shadow_free
netD_B = Discriminator()
netG_1 = Generator_S2F()  # shadow to shadow_free
netG_2 = Generator_F2S()  # shadow to shadow_free

if opt.cuda:
    netG_A2B.cuda()
    netD_B.cuda()
    netG_1.cuda()
    netG_2.cuda()

netG_A2B.apply(weights_init_normal)
netD_B.apply(weights_init_normal)
netG_1.apply(weights_init_normal)
netG_2.apply(weights_init_normal)


# Lossess
criterion_GAN = torch.nn.MSELoss()  # lsgan
# criterion_GAN = torch.nn.BCEWithLogitsLoss() #vanilla
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers

optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_1.parameters(), netG_2.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))

optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_A = Tensor(opt.batchSize, 3, opt.size, opt.size)
input_B = Tensor(opt.batchSize, 3, opt.size, opt.size)
input_C = Tensor(opt.batchSize, 1, opt.size, opt.size)
input_D = Tensor(opt.batchSize, 3, opt.size, opt.size)
input_E = Tensor(opt.batchSize, 3, opt.size, opt.size)
input_F = Tensor(opt.batchSize, 1, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_B_buffer = ReplayBuffer()

# Dataset loader
dataloader = DataLoader(ImageDataset(opt.dataroot, unaligned=True),
						batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

curr_iter = 0
G_losses_temp = 0
D_B_losses_temp = 0
G_losses = []
D_B_losses = []

open(opt.log_path, 'w').write(str(opt) + '\n\n')

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_nsr = Variable(input_A.copy_(batch['A']))#non shadow region:input;step1-gt
        random_sr = Variable(input_B.copy_(batch['B']))#random real shadow region:gan training
        mask = Variable(input_C.copy_(batch['C']))#nonshadow region mask:step2-input
        real_ns = Variable(input_D.copy_(batch['D']))#without real shadow region:step2-gt
        real_nsrs = Variable(input_E.copy_(batch['E']))#without nonshadow region and real shadow region-step2-input
        mask_dil = Variable(input_F.copy_(batch['F']))#without nonshadow region and real shadow region-step2-input
        
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(random_sr)
        loss_identity_B = criterion_identity(same_B, random_sr) * 5.0  # ||Gb(b)-b||1

        # GAN loss
        fake_B = netG_A2B(real_nsr)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)  # log(Db(Gb(a)))


        fake_nsr=netG_1(fake_B)
        loss_cycle=criterion_cycle(fake_nsr,real_nsr)
        
        output=netG_2(fake_nsr+real_nsrs,mask*2.0-1.0)
        loss_sr=criterion_identity(output,real_ns)
        
        loss_shadow=criterion_cycle(torch.cat(((output[:,0]+1.0)*mask_dil-1.0,output[:,1:]*mask_dil),1),torch.cat(((real_ns[:,0]+1.0)*mask_dil-1.0,real_ns[:,1:]*mask_dil),1))
        

        # Total loss
        loss_G = loss_identity_B + loss_GAN_A2B+loss_cycle+loss_sr+loss_shadow
        loss_G.backward()

        #G_losses.append(loss_G.item())
        G_losses_temp += loss_G.item()

        optimizer_G.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(random_sr)
        loss_D_real = criterion_GAN(pred_real, target_real)  # log(Db(b))

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)  # log(1-Db(G(a)))


        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        #D_B_losses.append(loss_D_B.item())
        D_B_losses_temp += loss_D_B.item()

        optimizer_D_B.step()
        ###################################

        curr_iter += 1

        if (i+1) % opt.iter_loss == 0:
            log = 'Epoch: %d, [iter %d], [loss_G %.5f], [loss_G_identity %.5f], [loss_G_GAN %.5f], [loss_D %.5f], [loss_shadow %.5f]' % \
                  (epoch, curr_iter, loss_G, loss_identity_B, loss_GAN_A2B,loss_D_B,loss_shadow)
            print(log)
            open(opt.log_path, 'a').write(log + '\n')

            G_losses.append(G_losses_temp / opt.iter_loss)
            D_B_losses.append(D_B_losses_temp / opt.iter_loss)
            G_losses_temp = 0
            D_B_losses_temp = 0

            avg_log = '[the last %d iters], [loss_G %.5f], [D_B_losses %.5f],' \
                      % (opt.iter_loss, G_losses[G_losses.__len__()-1], \
                         D_B_losses[D_B_losses.__len__()-1])
            print(avg_log)
            open(opt.log_path, 'a').write(avg_log + '\n')

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_B.step()


    if epoch>90:
        torch.save(netG_A2B.state_dict(), ('ckpt/netG_A2B_%d.pth' % (epoch + 1)))
        torch.save(netG_1.state_dict(), ('ckpt/netG_1_%d.pth' % (epoch + 1)))
        torch.save(netG_2.state_dict(), ('ckpt/netG_2_%d.pth' % (epoch + 1)))
        torch.save(netD_B.state_dict(), ('ckpt/netD_B_%d.pth' % (epoch+1)))

    print('Epoch:{}'.format(epoch))