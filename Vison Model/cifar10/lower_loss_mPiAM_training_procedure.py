"""Train CIFAR10 with PyTorch."""
from __future__ import print_function

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
from models import *
from adabound import AdaBound
from torch.optim import Adam, SGD
from optimizers import *
import numpy as np
import random

'''import sys



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
        
sys.stdout = Logger("/workspace0/ow120/DDAM/Graph_NN_case/cifar_10/Adabelief-Optimizer/PyTorch_Experiments/classification_cifar10/resnet_3.txt")'''

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--total_epoch', default=200, type=int, help='Total number of training epochs')
    parser.add_argument('--decay_epoch', default=150, type=int, help='Number of epochs to decay learning rate')
    parser.add_argument('--model', default='resnet', type=str, help='model',
                        choices=['resnet', 'densenet', 'vgg'])
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer',
                        choices=['sgd', 'adam', 'adamw', 'adabelief', 'yogi', 'msvag', 'radam', 'fromage', 'adabound',
                                 ])
    parser.add_argument('--run', default=0, type=int, help='number of runs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='learning rate')
    parser.add_argument('--final_lr', default=0.1, type=float,
                        help='final learning rate of AdaBound')
    parser.add_argument('--gamma', default=1e-3, type=float,
                        help='convergence speed term of AdaBound')

    parser.add_argument('--eps', default=1e-8, type=float, help='eps for var adam')

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--num_gpu', default=4, type=int, help='GPU to use for simulated parallel training.')
    parser.add_argument('--sigma_lr', default=0.08, type=float, help='sigma multiplier updating')
    parser.add_argument('--rho_lr', default=10000, type=float, help='rho multiplier updating')
    parser.add_argument('--beta_rmsprop', default=0.9, type=float, help='rmsprop coefficients beta_2')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--baseline_acc', default=0.9, type=float, help='baseline testing accuracy for saving model')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batchsize', type=int, default=128, help='batch size')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--reset', action = 'store_true',
                        help='whether reset optimizer at learning rate decay')
    return parser


def build_dataset(args):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True,
                                               num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader


def get_ckpt_name(model='resnet', optimizer='sgd', lr=0.1, final_lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, gamma=1e-3, eps=1e-8, weight_decay=5e-4,
                  reset = False, run = 0, weight_decouple = False, rectify = False):
    name = {
        'sgd': 'lr{}-momentum{}-wdecay{}-run{}'.format(lr, momentum,weight_decay, run),
        'adam': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2,weight_decay, eps, run),
        'fromage': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2,weight_decay, eps, run),
        'radam': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2,weight_decay, eps, run),
        'adamw': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2,weight_decay, eps, run),
        'adabelief': 'lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, eps, weight_decay, run),
        'adabound': 'lr{}-betas{}-{}-final_lr{}-gamma{}-wdecay{}-run{}'.format(lr, beta1, beta2, final_lr, gamma,weight_decay, run),
        'yogi':'lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, eps,weight_decay, run),
        'msvag': 'lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, eps,
                                                                    weight_decay, run),
    }[optimizer]
    return '{}-{}-{}-reset{}'.format(model, optimizer, name, str(reset))


def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(path)


def build_model(args, device, ckpt=None):
    print('==> Building model..')
    net = {
        'resnet': ResNet34,
        'densenet': DenseNet121,
        'vgg':vgg11,
    }[args.model]()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input: 3x32x32, Output: 32x32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x32x32
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling to reduce size, Output: 64x16x16
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  # Fully connected layer
        self.fc2 = nn.Linear(512, 10)  # Final output layer for 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))  # First conv layer
        x = self.pool(F.relu(self.conv2(x)))  # Second conv layer + Max pool
        x = x.view(-1, 64 * 16 * 16)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.fc2(x)  # Final output layer
        return x

def create_optimizer(args, model_params):
    args.optim = args.optim.lower()
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'fromage':
        return Fromage(model_params, args.lr)
    elif args.optim == 'radam':
        return RAdam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'adamw':
        return AdamW(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'adabelief':
        return AdaBelief(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'yogi':
        return Yogi(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'msvag':
        return MSVAG(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'adabound':
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay)
    else:
        print('Optimizer not found')


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


def generate_W_global(num_batches, W_n_list, P_n_list, tau_lr, alpha):
    W_n_avg = average_parameters(num_batches, W_n_list, alpha)
    P_n_avg = average_parameters(num_batches, P_n_list, alpha)
    for i in range(len(W_n_avg)):
        W_n_avg[i] = W_n_avg[i] + P_n_avg[i] / tau_lr
        W_n_avg[i].detach()

    #del P_n_avg
    #gc.collect()
    return W_n_avg

def train(net, epoch, device, data_loader, optimizer, criterion, args):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('train acc %.3f' % accuracy)

    return accuracy


def test(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('###############################################')
    print(' test acc %.3f' % accuracy)
    print('###############################################')

    return accuracy


def adjust_learning_rate(learning_rate, epoch, step_size, gamma, reset = False):

    if epoch % step_size==0 and epoch>4 and epoch <61:
    #if epoch % step_size==0 and epoch>4 and epoch <100:
        learning_rate *= gamma
        
    
    if epoch % 10==0 and epoch>61 and epoch <99:
        learning_rate *= gamma
    #if epoch % step_size==0 and epoch>61 and epoch <100:
    if epoch % 25==0 and epoch>99: # for resnet
        learning_rate *= 0.5
    #if epoch % step_size==0 and epoch>0 and epoch <100:
        #learning_rate *= gamma
    #if epoch % 25==0 and epoch>100 :
        #learning_rate *= 0.5
        
    return learning_rate

def main():
    parser = get_parser()
    args = parser.parse_args()

    train_loader, test_loader = build_dataset(args)
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

    ckpt_name = get_ckpt_name(model=args.model, optimizer=args.optim, lr=args.lr,
                              final_lr=args.final_lr, momentum=args.momentum,
                              beta1=args.beta1, beta2=args.beta2, gamma=args.gamma,
                              eps = args.eps,
                              reset=args.reset, run=args.run,
                              weight_decay = args.weight_decay)
    print('ckpt_name')
    if args.resume:
        ckpt = load_checkpoint(ckpt_name)
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']

        curve = os.path.join('curve', ckpt_name)     
        curve = torch.load(curve)
        train_accuracies = curve['train_acc']
        test_accuracies = curve['test_acc']
    else:
        ckpt = None
        best_acc = 0
        start_epoch = -1
        train_accuracies = []
        test_accuracies = []

    model = build_model(args, device, ckpt=ckpt)
    #model = CNN()
    #model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    W_n_0 = [param.clone().detach().requires_grad_(True) for param in model.parameters()]    
    W_b_initial = [[param.clone() for param in W_n_0] for _ in range(args.num_gpu)]
    P_b_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(args.num_gpu)]
    accumulators_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(args.num_gpu)]
    velocities_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(args.num_gpu)]
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.1,
    #                                      last_epoch=start_epoch)
    
    sigma_lr = args.sigma_lr
    alpha_b = [1/args.num_gpu for _ in range(args.num_gpu)]
    print('alpha_b is: ', alpha_b)
    W_global = generate_W_global(args.num_gpu, W_b_initial, P_b_initial, sigma_lr, alpha_b)
    zero_grad(model.parameters())
    #optimizer.zero_grad()
    learning_rate_current = 1/(args.sigma_lr + args.rho_lr)
    #learning_rate_current = args.lr
    sigma_lr_current = args.sigma_lr
    updated_iteration = 1.0
    epoch_loss = 0
    best_acc = args.baseline_acc
    

    for epoch in range(start_epoch + 1, args.total_epoch):
        start = time.time()
        #scheduler.step()
        
        update_count = 0
        total_train_loss = 0
        learning_rate_current = adjust_learning_rate(learning_rate_current, epoch, step_size=args.decay_epoch, gamma=args.lr_gamma, reset = args.reset)
        sigma_lr_current = ((1/(args.sigma_lr + args.rho_lr))/learning_rate_current) * args.sigma_lr
        rho_lr_current = 1/learning_rate_current - sigma_lr_current
        #sigma_lr_current = (args.lr/learning_rate_current) * args.sigma_lr
        print('Current LR is:', learning_rate_current)
        print('Current sigma_lr is:', sigma_lr_current)
        #train_acc = train(net, epoch, device, train_loader, optimizer, criterion, args)
        
        for iter_idx, (images, target) in enumerate(train_loader):
            sub_batch_size = images.size(0) // args.num_gpu
            alpha_b = []
           
                    
            for sb in range(args.num_gpu):
                with torch.no_grad():  # Disable gradient tracking
                    for param, w in zip(model.parameters(), W_global):
                        param.copy_(w)
                if sb == args.num_gpu - 1:
                    images_sub = images[sub_batch_size*sb:].to(device)
                    target_sub = target[sub_batch_size*sb:].to(device)
                else:
                    images_sub = images[sub_batch_size*sb:sub_batch_size*(sb+1)].to(device)
                    target_sub = target[sub_batch_size*sb:sub_batch_size*(sb+1)].to(device)
                    
                #print('The number of images per sub-batch:', len(images_sub))
                W_n = W_b_initial[sb]
                P_n = P_b_initial[sb]
                accumulators = accumulators_initial[sb]
                velocities = velocities_initial[sb]
                alpha_b.append(images_sub.size(0)/images.size(0))
                
                
                output = model(images_sub)
                #l2_penalty = sum(torch.sum(param ** 2) for param in model.parameters())
                loss = criterion(output, target_sub)
                total_train_loss += loss.item()
                
                #loss+= args.weight_decay * l2_penalty
                zero_grad(model.parameters())
                #optimizer.zero_grad()
                loss.backward()
                gradients = [param.grad + args.weight_decay*param for param in model.parameters()]
                
                
                with torch.no_grad():

                    for i, (param_wn, param_pn, gradient, param_wg, accumulator, velocity) in enumerate(zip(W_n, P_n, gradients, W_global, accumulators, velocities)):
                        #velocity.mul_(args.beta1).add_((1 - args.beta1) * (gradient + param_pn))
                        accumulator.mul_(args.beta_rmsprop).add_((1 - args.beta_rmsprop) * (gradient + param_pn).pow(2))
                        #accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) * gradient.pow(2))
                        #accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) *  param_pn.pow(2))
                        
                        
                        bias_correction2 = 1 - args.beta_rmsprop** updated_iteration                        
                        corrected_accumulator = accumulator / (bias_correction2)
                        #bias_correction1 = 1 - args.beta1** updated_iteration                        
                        #corrected_velocity= velocity / (bias_correction1)
                        
                        
                        
                        
                        #delta = param_wg - learning_rate_current * corrected_velocity/(torch.sqrt(corrected_accumulator) + args.eps)
                        #delta = param_wg - learning_rate_current * (gradient+ param_pn)/(torch.sqrt(corrected_accumulator) + args.eps)
                        delta = param_wg -  (gradient+ param_pn)/(sigma_lr_current + rho_lr_current*(torch.sqrt(corrected_accumulator) + args.eps))
                        
                        param_wn.copy_(delta.detach())
                        param_pn.add_(sigma_lr_current * (param_wn - param_wg))
                        
                #zero_grad(model.parameters())
                del loss
                del output
                
            updated_iteration += 1
                    
                
            with torch.no_grad():
                #W_global = generate_W_global(num_gpu, W_b_initial, P_b_initial, sigma_lr, alpha_b_n[(update_count-num_gpu):update_count])
                W_global = generate_W_global(args.num_gpu, W_b_initial, P_b_initial, sigma_lr_current, alpha_b)
                for param, w in zip(model.parameters(), W_global):
                    param.copy_(w)
        
    

            #if iter_idx %10 == 0 and iter_idx >0:
                #print(f"{iter_idx} in {len(train_loader)} updating iteration in {epoch} epoch Average training loss: {avg_batch_loss}")
        
        print(f"updating iteration in {epoch} epoch Average training loss: {total_train_loss/(len(train_loader)*args.num_gpu)}")
        train_acc = total_train_loss/(len(train_loader)*args.num_gpu)
        test_acc = test(model, device, test_loader, criterion)
        end = time.time()
        print('Time: {}'.format(end-start))
        model.train()
        #test_acc = 0
        if test_acc > best_acc:
            best_acc = test_acc
            #torch.save(model.state_dict(), "resnet.pth")
            torch.save(model.state_dict(), "/workspace0/ow120/DDAM/Graph_NN_case/cifar_10/Adabelief-Optimizer/PyTorch_Experiments/classification_cifar10/ResNet_best_model/model.pth")
            #torch.save(model.state_dict(), "/workspace1/ow120/DDAM/Graph_NN_case/cifar_10/Adabelief-Optimizer/PyTorch_Experiments/classification_cifar10/VGG_best_model/model.pth")
            #torch.save(model.state_dict(), "/workspace1/ow120/DDAM/Graph_NN_case/cifar_10/Adabelief-Optimizer/PyTorch_Experiments/classification_cifar10/DenseNet_best_model/model_1.pth")
            

        # Save checkpoint.
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, os.path.join('checkpoint', ckpt_name))
            best_acc = test_acc

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        if not os.path.isdir('curve_lower_loss'):
            os.mkdir('curve_lower_loss')
        torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                   os.path.join('curve_lower_loss', ckpt_name))
    print('The best accuracy is:', max(test_accuracies))

if __name__ == '__main__':
    main()
