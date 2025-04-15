from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from optimizers import *
from adabound import AdaBound
from torch.optim import SGD, Adam
from fid_score import calculate_fid_given_paths
import pandas as pd
import numpy as np
import sys
import random
from inception import InceptionV3
from torchvision import transforms
from PIL import Image

#random.seed(42)
np.random.seed(42)
#torch.manual_seed(42)
#if torch.cuda.is_available():
    #torch.cuda.manual_seed_all(42)

class Logger(object):
    def __init__(self, fileN="record.txt"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()                 # flush the file after each write
    def flush(self):
        self.log.flush()
        
sys.stdout = Logger("/workspace1/ow120/DDAM/Graph_NN_case/cifar_10/Adabelief-Optimizer/PyTorch_Experiments/wgan/wgan_1.txt")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='cifar10', help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False, default='./',help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=3)
parser.add_argument('--partial', default=1.0/4.0, type=float)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer')
parser.add_argument('--beta2', default=0.999, type=float, help='Beta2')
parser.add_argument('--eps',default=1e-8, type=float, help='eps')
parser.add_argument('--final_lr', default=1e-2, type=float, help='final_lr')
parser.add_argument('--Train', action = 'store_true')
parser.add_argument('--run', default=0, type=int, help='runs')
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument('--num_gpu', default=4, type=int, help='GPU to use for simulated parallel training.')
parser.add_argument('--sigma_lr_generator', default=0.05, type=float, help='sigma multiplier updating')
parser.add_argument('--rho_lr_generator', default=10000, type=float, help='rho multiplier updating')
parser.add_argument('--sigma_lr_discriminator', default=0.05, type=float, help='sigma multiplier updating')
parser.add_argument('--rho_lr_discriminator', default=10000, type=float, help='rho multiplier updating')
parser.add_argument('--l2_lambda', default=1e-5, type=float, help='weight decay for global w')
parser.add_argument('--beta_rmsprop', default=0.999, type=float, help='rmsprop coefficients beta_2')

opt = parser.parse_args()
opt.outf = 'SPiAM' + '-wgan_1' + '-betas{}-{}'.format(opt.beta1, opt.beta2) + '-eps{}'.format(opt.eps) \
           + '-final-lr{}'.format(opt.final_lr) + '-run{}'.format(str(opt.run)) + '-clip-{}'.format(opt.clip_value)
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif opt.dataset == 'lsun':
    classes = [ c + '_train' for c in opt.classes.split(',')]
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc=1

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

#device = torch.device("cuda:{}".format(os.environ['CUDA_VISIBLE_DEVICES']) if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def zero_grad(params):
    """
    Zeroes out gradients for the given parameters.
    """
    for param in params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
            
            
def average_parameters(num_train_env, list_vars, list_alpha):
    sum_vars = [torch.zeros_like(var) for var in list_vars[0]]
    for i in range(num_train_env):
        W_n = list_vars[i]
        alpha = list_alpha[i]
        sum_vars = [sum_ + alpha*update for sum_, update in zip(sum_vars, W_n)]
    return sum_vars


def generate_W_global(num_batches, W_n_list, P_n_list, tau_lr, alpha, l2_lambda):
    W_n_avg = average_parameters(num_batches, W_n_list, alpha)
    P_n_avg = average_parameters(num_batches, P_n_list, alpha)
    for i in range(len(W_n_avg)):
        W_n_avg[i] = (tau_lr*W_n_avg[i] + P_n_avg[i]) / (tau_lr + l2_lambda)
        W_n_avg[i].detach()

    #del P_n_avg
    #gc.collect()
    return W_n_avg


def load_images_from_folder(folder, image_size=(299, 299)):
    """
    Load all images from a folder into a torch.Tensor suitable for FID computation.
    
    Params:
    -- folder      : Path to the folder containing images.
    -- image_size  : Target size to resize images (defaults to 299x299 for InceptionV3).
    
    Returns:
    -- images      : Torch tensor of shape (N, C, H, W), where N is the number of images.
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()  # Converts to range [0, 1]
    ])

    images = []
    for filename in sorted(os.listdir(folder)):  # Sort to ensure consistent order
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("RGB")  # Ensure 3 channels (RGB)
            img_tensor = transform(img)
            images.append(img_tensor)
    
    images = torch.stack(images)  # Stack into a single tensor of shape (N, C, H, W)
    return images


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).cuda()
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).cuda()
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)


W_n_generator = [param.clone().detach().requires_grad_(True) for param in netG.parameters()]    
W_b_initial_generator = [[param.clone() for param in W_n_generator] for _ in range(opt.num_gpu)]
P_b_initial_generator = [[torch.zeros_like(param) for param in W_n_generator] for _ in range(opt.num_gpu)]
accumulators_initial_generator = [[torch.zeros_like(param) for param in W_n_generator] for _ in range(opt.num_gpu)]


W_n_discriminator = [param.clone().detach().requires_grad_(True) for param in netD.parameters()]    
W_b_initial_discriminator = [[param.clone() for param in W_n_discriminator] for _ in range(opt.num_gpu)]
P_b_initial_discriminator = [[torch.zeros_like(param) for param in W_n_discriminator] for _ in range(opt.num_gpu)]
accumulators_initial_discriminator = [[torch.zeros_like(param) for param in W_n_discriminator] for _ in range(opt.num_gpu)]

############# initialization #############
sigma_lr_generator = opt.sigma_lr_generator
sigma_lr_discriminator = opt.sigma_lr_discriminator
rho_lr_generator = 1/opt.lr - sigma_lr_generator
rho_lr_discriminator = 1/opt.lr - sigma_lr_discriminator
alpha_b = [1/opt.num_gpu for _ in range(opt.num_gpu)]
print('alpha_b is: ', alpha_b)
updated_iteration_discriminator = 1.0
updated_iteration_generator = 1.0
W_global_generator = generate_W_global(opt.num_gpu, W_b_initial_generator, P_b_initial_generator, sigma_lr_generator, alpha_b, opt.l2_lambda)
W_global_discriminator = generate_W_global(opt.num_gpu, W_b_initial_discriminator, P_b_initial_discriminator, sigma_lr_discriminator, alpha_b, opt.l2_lambda)
zero_grad(netG.parameters())
zero_grad(netD.parameters())


criterion = nn.BCELoss()


fixed_noise = torch.randn(opt.batchSize, nz, 1, 1).cuda()#, device=device)
real_label = 1
fake_label = 0

# setup optimizer
# setup optimizer
opt.optimizer = opt.optimizer.lower()
if opt.optimizer == 'adam':
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
elif opt.optimizer == 'adamw':
    optimizerD = AdamW(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    optimizerG = AdamW(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
elif opt.optimizer == 'fromage':
    optimizerD = Fromage(netD.parameters(), lr=opt.lr)
    optimizerG = Fromage(netG.parameters(), lr=opt.lr)
elif opt.optimizer == 'adabelief':
    optimizerD = AdaBelief(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    optimizerG = AdaBelief(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
elif opt.optimizer == 'sgd':
    optimizerD = torch.optim.SGD(netD.parameters(), lr=opt.lr)
    optimizerG = torch.optim.SGD(netG.parameters(), lr=opt.lr)
elif opt.optimizer == 'rmsprop':
    optimizerD = torch.optim.RMSprop(netD.parameters(), lr=opt.lr)
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.lr)
elif opt.optimizer == 'adabound':
    optimizerD = AdaBound(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps,
                          final_lr=opt.final_lr)
    optimizerG = AdaBound(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps,
                          final_lr = opt.final_lr)
elif opt.optimizer == 'yogi':
    optimizerD = Yogi(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),eps=opt.eps)
    optimizerG = Yogi(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),eps=opt.eps)
elif opt.optimizer == 'radam':
    optimizerD = RAdam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),eps=opt.eps)
    optimizerG = RAdam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),eps=opt.eps)
elif opt.optimizer == 'msvag':
    optimizerD = MSVAG(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimizerG = MSVAG(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

# convert all training data into png format
real_folder = 'all_real_imgs'
if not os.path.exists(real_folder):
    os.mkdir(real_folder)
    for i in range(len(dataset)):
        vutils.save_image(dataset[i][0], real_folder + '/{}.png'.format(i), normalize=True)

fake_folder = 'all_fake_imgs' + opt.outf
if not os.path.exists(fake_folder):
    os.mkdir(fake_folder)

FIDs = []
fake_images_number = 1000

print(opt.Train)

### without loading images from folder
'''dims=2048
inception_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]]).cuda()
inception_model.eval()'''
#print('Prepare to load real images')
#real_images = load_images_from_folder(real_folder)
#mu_real, sigma_real = _compute_statistics_of_path(real_folder, inception_model, opt.batchSize//2, dims, True)
#print('Finish loading real images!!')
if opt.Train == True:

    for epoch in range(opt.niter):
        print('Epoch {}'.format(epoch))
        
        for i, data in enumerate(dataloader, 0):
            sub_batch_size = data[0].size(0) // opt.num_gpu #data is a list, data[0] contains batch of images
            alpha_b = []
            
            if i % opt.n_critic == 0: ### save the noise during training generators
                noise_record = [] 
            
            for sb in range(opt.num_gpu):
                with torch.no_grad():  # Disable gradient tracking
                    for param, w in zip(netD.parameters(), W_global_discriminator):
                        param.copy_(w)
                if sb == opt.num_gpu - 1:
                    data_sub = data[0][sub_batch_size*sb:]
                else:
                    data_sub = data[0][sub_batch_size*sb:sub_batch_size*(sb+1)]
                    
                W_n_discriminator = W_b_initial_discriminator[sb]
                P_n_discriminator = P_b_initial_discriminator[sb]
                accumulators_discriminator = accumulators_initial_discriminator[sb]
                alpha_b.append(data_sub.size(0)/data[0].size(0))
            
            
                
                # Configure input
                real_imgs = data_sub.cuda()#Variable(imgs.type(Tensor))
    
                # ---------------------
                #  Train Discriminator
                # ---------------------
    
                #optimizerD.zero_grad()
    
                # Sample noise as netG input
                #z = torch.randn(sub_batch_size, nz, 1, 1).cuda()#, device=device)#Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
                z = torch.randn(data_sub.size(0), nz, 1, 1)
                if i % opt.n_critic == 0: ### save the noise during training generators
                    noise_record.append(z)
                
    
                # Generate a batch of images
                #fake_imgs = netG(z).detach()
                fake_imgs = netG(z.cuda()).detach()
                # Adversarial loss
                loss_D = -torch.mean(netD(real_imgs)) + torch.mean(netD(fake_imgs))
    
                zero_grad(netD.parameters())
                #optimizerD.zero_grad()
                loss_D.backward()
                #optimizerD.step()
                gradients_discriminator = [param.grad for param in netD.parameters()]
                
                
                with torch.no_grad():

                    for ii, (param_wn, param_pn, gradient, param_wg, accumulator) in enumerate(zip(W_n_discriminator, P_n_discriminator, gradients_discriminator, W_global_discriminator, accumulators_discriminator)):
                        #velocity.mul_(opt.beta1).add_((1 - opt.beta1) * (gradient + param_pn))
                        accumulator.mul_(opt.beta_rmsprop).add_((1 - opt.beta_rmsprop) * (gradient + param_pn).pow(2))
                        #accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) * gradient.pow(2))
                        #accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) *  param_pn.pow(2))
                        
                        
                        bias_correction2 = 1 - opt.beta_rmsprop** updated_iteration_discriminator                        
                        corrected_accumulator = accumulator / (bias_correction2)
                        #bias_correction1 = 1 - opt.beta1** updated_iteration                        
                        #corrected_velocity= velocity / (bias_correction1)
                        
                        
                        
                        
                        #delta = param_wg - learning_rate_current * corrected_velocity/(torch.sqrt(corrected_accumulator) + opt.eps)
                        #delta = param_wg - opt.lr * (gradient+ param_pn)/(torch.sqrt(corrected_accumulator) + opt.eps)
                        delta = param_wg -  (gradient+ param_pn)/(sigma_lr_discriminator + rho_lr_discriminator*(torch.sqrt(corrected_accumulator) + opt.eps))
                        
                        param_wn.copy_(delta.detach())
                        param_pn.add_(sigma_lr_discriminator* (param_wn - param_wg))
                        

                        
                del loss_D

            
            # Clip weights of netD
            
            with torch.no_grad():
                W_global_discriminator = generate_W_global(opt.num_gpu, W_b_initial_discriminator, P_b_initial_discriminator, sigma_lr_discriminator, alpha_b, opt.l2_lambda)
                for param, w in zip(netD.parameters(), W_global_discriminator):
                #for param, w in zip(netD.parameters(), W_b_initial_discriminator[0]):
                    param.copy_(w)
                    
                    
                    
            updated_iteration_discriminator += 1
            #for p in netD.parameters():
                #p.data.clamp_(-opt.clip_value, opt.clip_value)
                
            with torch.no_grad():
                for param, w in zip(netD.parameters(), W_global_discriminator):
                    param.data.clamp_(-opt.clip_value, opt.clip_value)
                    w.copy_(param)

            # Train the netG every n_critic iterations
            if i % opt.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                for sb in range(opt.num_gpu):
                    with torch.no_grad():  # Disable gradient tracking
                        for param, w in zip(netG.parameters(), W_global_generator):
                            param.copy_(w)
                        
                    W_n_generator = W_b_initial_generator[sb]
                    P_n_generator = P_b_initial_generator[sb]
                    accumulators_generator = accumulators_initial_generator[sb]
                    z = noise_record[sb]

                    #optimizerG.zero_grad()
    
                    # Generate a batch of images
                    gen_imgs = netG(z.cuda())
                    # Adversarial loss
                    loss_G = -torch.mean(netD(gen_imgs))
    
                    zero_grad(netG.parameters())
                    #optimizerG.zero_grad()
                    loss_G.backward()
                    #optimizerG.step()
                    gradients_generator = [param.grad for param in netG.parameters()]
                    with torch.no_grad():

                        for ii, (param_wn, param_pn, gradient, param_wg, accumulator) in enumerate(zip(W_n_generator, P_n_generator, gradients_generator, W_global_generator, accumulators_generator)):
                            #velocity.mul_(opt.beta1).add_((1 - opt.beta1) * (gradient + param_pn))
                            accumulator.mul_(opt.beta_rmsprop).add_((1 - opt.beta_rmsprop) * (gradient + param_pn).pow(2))
                            #accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) * gradient.pow(2))
                            #accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) *  param_pn.pow(2))
                            
                            
                            bias_correction2 = 1 - opt.beta_rmsprop** updated_iteration_generator                        
                            corrected_accumulator = accumulator / (bias_correction2)
                            #bias_correction1 = 1 - opt.beta1** updated_iteration                        
                            #corrected_velocity= velocity / (bias_correction1)
                            
                            
                            
                            
                            #delta = param_wg - learning_rate_current * corrected_velocity/(torch.sqrt(corrected_accumulator) + opt.eps)
                            #delta = param_wg - opt.lr * (gradient+ param_pn)/(torch.sqrt(corrected_accumulator) + opt.eps)
                            delta = param_wg -  (gradient+ param_pn)/(sigma_lr_generator + rho_lr_generator*(torch.sqrt(corrected_accumulator) + opt.eps))
                            
                            param_wn.copy_(delta.detach())
                            param_pn.add_(sigma_lr_generator * (param_wn - param_wg))
                            
                    del loss_G
                    #optimizerG.step()
                    
                with torch.no_grad():
                    #W_global = generate_W_global(num_gpu, W_b_initial, P_b_initial, sigma_lr, alpha_b_n[(update_count-num_gpu):update_count])
                    W_global_generator = generate_W_global(opt.num_gpu, W_b_initial_generator, P_b_initial_generator, sigma_lr_generator, alpha_b, opt.l2_lambda)
                    for param, w in zip(netG.parameters(), W_global_generator):
                        param.copy_(w)
                        
                updated_iteration_generator += 1



            

        if (epoch+1) %10 == 0:
            fake_folder_temp = fake_folder +'/epoch_'+ str(epoch)
            if not os.path.exists(fake_folder_temp):
                os.mkdir(fake_folder_temp)
            batch_size = opt.batchSize
            netG.eval()
            for i in range(int(fake_images_number/10)):
                noise = torch.randn(batch_size, nz, 1, 1).cuda()
                fake = netG(noise)
                for j in range(fake.shape[0]):
                    vutils.save_image(fake.detach()[j,...], fake_folder_temp + '/{}.png'.format(j + i * batch_size), normalize=True)
            netG.train()
        
            # calculate FID score
            fid_value = calculate_fid_given_paths([real_folder, fake_folder_temp],
                                                  opt.batchSize//2,
                                                 True)
            '''fid_value = calculate_fid_given_paths([real_folder, fake_folder_temp],
                                                  opt.batchSize//2,
                                                 True, inception_model)'''
            ### The problem is this stpe within 'calculate_frechet_distance' is too slow, the slow reason is not due to loading images from disk

            print('FID: {}'.format(fid_value))
            FIDs.append(fid_value)



        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

if True:
    batch_size = opt.batchSize
    netG.load_state_dict(torch.load('%s/netG_epoch_%d.pth' % (opt.outf, opt.niter-1)))

    # test netG, and calculate FID score
    netG.eval()
    for i in range(fake_images_number):
        noise = torch.randn(batch_size, nz, 1, 1).cuda()
        fake = netG(noise)
        for j in range(fake.shape[0]):
            vutils.save_image(fake.detach()[j,...], fake_folder + '/{}.png'.format(j + i * batch_size), normalize=True)
    netG.train()

    # calculate FID score
    '''fid_value = calculate_fid_given_paths([real_folder, fake_folder],
                                          opt.batchSize//2,
                                         True, inception_model)'''
                                         
    fid_value = calculate_fid_given_paths([real_folder, fake_folder],
                                                  opt.batchSize//2,
                                                 True)
    FIDs.append(fid_value)

    print('FID: {}'.format(fid_value))

    df = pd.DataFrame(FIDs)
    df.to_csv('FID_{}.csv'.format(opt.outf))

