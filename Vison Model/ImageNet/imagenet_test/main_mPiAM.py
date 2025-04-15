import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from adabound import AdaBound
from shufflenet import *
from senet import *
import pandas as pd
from SGD_GC import SGD_GC
from adabelief_pytorch import AdaBelief
from AdamW import AdamW
from RAdam import RAdam
#from MSVAG import MSVAG
import sys

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
        
sys.stdout = Logger("/workspace1/ow120/DDAM/Graph_NN_case/cifar_10/Adabelief-Optimizer/PyTorch_Experiments/imagenet/imagenet_2.txt")


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',default='/workspace1/ow120/DDAM/Imagenet/dataset/imagenet/',#'/data1/ILSVRC2012',#
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--optimizer', default='adabelief', type=str)
parser.add_argument('--centralize', default=False, dest='centralize', action = 'store_true')
parser.add_argument('--reset', default=False, dest='reset', action = 'store_true')
parser.add_argument('--warmup', default=False, dest='warmup', action = 'store_true')
parser.add_argument('--reset_resume_optim', default=False, dest='reset_resume_optim', action = 'store_true')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--when', nargs='+', type=int, default=[-1])
parser.add_argument('--decay_epoch', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--fixed_decay', action='store_true')
parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--num_gpu', default=4, type=int, help='GPU to use for simulated parallel training.')
parser.add_argument('--sigma_lr', default=0.005, type=float, help='sigma multiplier updating')
parser.add_argument('--rho_lr', default=10000, type=float, help='rho multiplier updating')
parser.add_argument('--beta_rmsprop', default=0.999, type=float, help='rmsprop coefficients beta_2')
parser.add_argument('--l1_decay', default=0.0, type=float)
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--eps', default=1e-8, type=float, help='eps in Adabelief')
parser.add_argument('--decay_rate', default=0.1, type=float, help='learning rate decay')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1 in Adabelief')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 in Adabelief')
parser.add_argument('--weight_decouple', default=True, type=bool, help='Weight decouple in Adabelief')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0

def average_parameters(num_train_env, list_vars, list_alpha):
    sum_vars = [torch.zeros_like(var) for var in list_vars[0]]
    for i in range(num_train_env):
        W_n = list_vars[i]
        alpha = list_alpha[i]
        sum_vars = [sum_ + alpha*update for sum_, update in zip(sum_vars, W_n)]
    return sum_vars


def generate_W_global(num_batches, W_n_list, P_n_list, tau_lr, alpha):
    W_n_avg = average_parameters(num_batches, W_n_list, alpha)
    P_n_avg = average_parameters(num_batches, P_n_list, alpha)
    for i in range(len(W_n_avg)):
        W_n_avg[i] = W_n_avg[i] + P_n_avg[i] / tau_lr
        W_n_avg[i].detach()

    #del P_n_avg
    #gc.collect()
    return W_n_avg

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

# Seems to speed up training by ~2%
class DataPrefetcher():
    def __init__(self, loader, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)
    
    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break

def zero_grad(params):
    """
    Zeroes out gradients for the given parameters.
    """
    for param in params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
            

def main_worker(gpu, ngpus_per_node, args):
    filename = 'model-{}-optimizer-{}-lr-{}-epochs-{}-decay-epoch-{}-eps{}-beta1{}-beta2{}-centralize-{}-reset{}-start-epoch-{}-l2-decay{}-l1-decay{}-batch-{}-warmup-{}-fixed-decay-{}'.format(args.arch, args.optimizer, args.lr, args.epochs, args.when, args.eps, args.beta1, args.beta2, args.centralize, args.reset, args.start_epoch, args.weight_decay, args.l1_decay, args.batch_size, args.warmup, args.fixed_decay)

    print(filename)

    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch == 'shufflenet_v2_x0_5':
           model = shufflenet_v2_x0_5(pretrained=False)
        elif args.arch == 'se_resnet18':
           model = se_resnet18()
        else:
           model = models.__dict__[args.arch]()
    '''
    model.half()  # convert to half precision
    for layer in model.modules():
      if isinstance(layer, nn.BatchNorm2d):
        layer.float()
    '''
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optimizer == 'sgd' and (not args.centralize):
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd' and args.centralize:
        optimizer = SGD_GC(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(), args.lr, eps=args.eps, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'adabelief':
        optimizer = AdaBelief(model.parameters(), args.lr, eps=args.eps, betas=(args.beta1, args.beta2), weight_decouple = args.weight_decouple, 
                              weight_decay = args.weight_decay, fixed_decay = args.fixed_decay, rectify=False)
    elif args.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), args.lr, eps=args.eps, betas=(args.beta1, args.beta2), weight_decay = args.weight_decay)
    #elif args.optimizer == 'msvag':
    #    optimizer = MSVAG(model.parameters(), args.lr, eps=args.eps, betas=(args.beta1, args.beta2), weight_decay = args.weight_decay)
    else:
        print('Optimizer not found')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            
            if args.start_epoch is None:
                args.start_epoch = checkpoint['epoch'] + 1
                df = pd.read_csv(filename+'.csv')
                train1, train5, test1, test5 = df['train1'].tolist(), df['train5'].tolist(), df['test1'].tolist(), df['test5'].tolist()
            else: # if specify start epoch, and resume from checkpoint, not resume previous accuracy curves
                train1, train5, test1, test5 = [], [], [], []
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])

            if not args.reset_resume_optim:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        if args.start_epoch is None:
            args.start_epoch = 0
        train1, train5, test1, test5 = [], [], [], []

    cudnn.benchmark = True
    
    # System Definition
    W_n_0 = [param.clone().detach().requires_grad_(True) for param in model.parameters()]    
    W_b_initial = [[param.clone() for param in W_n_0] for _ in range(args.num_gpu)]
    P_b_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(args.num_gpu)]
    accumulators_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(args.num_gpu)]
    velocities_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(args.num_gpu)]

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    train_loader, val_loader = DataPrefetcher(train_loader), DataPrefetcher(val_loader)
    alpha_b = [1/args.num_gpu for _ in range(args.num_gpu)]
    sigma_lr = args.sigma_lr
    W_global = generate_W_global(args.num_gpu, W_b_initial, P_b_initial, sigma_lr, alpha_b)
    zero_grad(model.parameters())
    learning_rate_current = args.lr
    updated_iteration = 1.0
    

    
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        update_count = 0
        total_train_loss = 0
        learning_rate_current = adjust_learning_rate(learning_rate_current, epoch, args)
        sigma_lr_current = (args.lr/learning_rate_current) * args.sigma_lr
        rho_lr_current = 1/learning_rate_current - sigma_lr_current
        
        print('Current LR is:', learning_rate_current)
        print('Current sigma_lr is:', sigma_lr_current)

        model.train()
        
        losses_train = AverageMeter('Loss', ':.4e')
        top1_train = AverageMeter('Acc@1', ':6.2f')
        top5_train = AverageMeter('Acc@5', ':6.2f')
        progress_train = ProgressMeter(
            len(train_loader),
            [losses_train, top1_train, top5_train],
            prefix="Epoch: [{}]".format(epoch))
            
        for iter_idx, (images, target) in enumerate(train_loader):
            sub_batch_size = images.size(0) // args.num_gpu
            alpha_b = []
            batch_loss = 0
            batch_acc1 = 0
            batch_acc5 = 0
            
            for sb in range(args.num_gpu):
                with torch.no_grad():  # Disable gradient tracking
                    for param, w in zip(model.parameters(), W_global):
                        param.copy_(w)
                if sb == args.num_gpu - 1:
                    images_sub = images[sub_batch_size*sb:].cuda(args.gpu, non_blocking=True)
                    target_sub = target[sub_batch_size*sb:].cuda(args.gpu, non_blocking=True)
                else:
                    images_sub = images[sub_batch_size*sb:sub_batch_size*(sb+1)].cuda(args.gpu, non_blocking=True)
                    target_sub = target[sub_batch_size*sb:sub_batch_size*(sb+1)].cuda(args.gpu, non_blocking=True)
                    
                #print('The number of images per sub-batch:', len(images_sub))
                W_n = W_b_initial[sb]
                P_n = P_b_initial[sb]
                accumulators = accumulators_initial[sb]
                velocities = velocities_initial[sb]
                alpha_b.append(images_sub.size(0)/images.size(0))
                
                
                output = model(images_sub)
                acc1_sub, acc5_sub = accuracy(output, target_sub, topk=(1, 5))
                #l2_penalty = sum(torch.sum(param ** 2) for param in model.parameters())
                loss = criterion(output, target_sub)
                total_train_loss += loss.item()
                batch_loss += loss.item()
                batch_acc1 += acc1_sub[0]
                batch_acc5 += acc5_sub[0]
                
                #loss+= args.weight_decay * l2_penalty 
                zero_grad(model.parameters())
                loss.backward()
                #gradients = [param.grad + args.weight_decay*param for param in model.parameters()]
                gradients = [param.grad for param in model.parameters()]
                
                
                with torch.no_grad():

                    for i, (param_wn, param_pn, gradient, param_wg, accumulator, velocity) in enumerate(zip(W_n, P_n, gradients, W_global, accumulators, velocities)):
                        #velocity.mul_(args.beta1).add_((1 - args.beta1) * (gradient + param_pn))
                        accumulator.mul_(args.beta_rmsprop).add_((1 - args.beta_rmsprop) * (gradient + param_pn).pow(2))
                        #accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) * gradient.pow(2))
                        #accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) *  param_pn.pow(2))
                        
                        
                        bias_correction2 = 1 - args.beta_rmsprop** updated_iteration                        
                        corrected_accumulator = accumulator / (bias_correction2)
                        delta = param_wg -  (gradient+ param_pn)/(sigma_lr_current + rho_lr_current*(torch.sqrt(corrected_accumulator) + args.eps))
                        
                        param_wn.copy_(delta.detach())
                        param_pn.add_(sigma_lr_current * (param_wn - param_wg))
                        
                #zero_grad(model.parameters())
                del loss
                del output
                
            updated_iteration += 1
            
            losses_train.update(batch_loss/args.num_gpu, images.size(0))
            top1_train.update(batch_acc1/args.num_gpu, images.size(0))
            top5_train.update(batch_acc5/args.num_gpu, images.size(0))
            if iter_idx % args.print_freq == 0:
                progress_train.display(iter_idx)
                    
                
            with torch.no_grad():
                W_global = generate_W_global(args.num_gpu, W_b_initial, P_b_initial, sigma_lr_current, alpha_b)
                for param, w in zip(model.parameters(), W_global):
                    param.copy_(w)
        
    

        
        print(f"updating iteration in {epoch} epoch Average training loss: {total_train_loss/(len(train_loader)*args.num_gpu)}")
            
        
        

        # evaluate on validation set
        acc1, _test5 = validate(val_loader, model, criterion, args)

        #train1.append(_train1.data.cpu().numpy())
        #train5.append(_train5.data.cpu().numpy())
        test1.append(acc1.data.cpu().numpy())
        test5.append(_test5.data.cpu().numpy())
        results = {}
        #results['train1'] = train1
        #results['train5'] = train5
        results['test1'] = test1
        results['test5'] = test5
        df = pd.DataFrame(data = results)
        df.to_csv(filename+'.csv')

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename = filename, epoch=epoch, decay_epoch = args.decay_epoch)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
     
        #images=images.half()
        #targets=target.half()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.detach().item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # warmup learning rate
        if args.warmup and epoch % args.decay_epoch == 0:
            print('Warmup')
            lr = args.lr * (0.1 ** (epoch // args.decay_epoch))
            lr_tmp = lr / float(len(train_loader)) * float(i)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_tmp
            

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            #target = target.half()
            #images = images.half()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', epoch = 0, decay_epoch=30):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'_model_best.pth.tar')
    if (epoch + 1) % decay_epoch == 0:
        torch.save(state, '{}-epoch-{}'.format(filename, epoch))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(lr_current, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in args.when:
        lr_current *= args.decay_rate
        #lr_current *= 0.1
        
    else:
        lr_current = lr_current
        
    return lr_current
        
    
    #lr = args.lr * (0.1 ** (epoch // args.decay_epoch))
    #if args.reset and (epoch % args.decay_epoch) == 0:
    #    print('Reset optimizer')
    #    optimizer.reset()
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':  
    main()
