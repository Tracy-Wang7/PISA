import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import os
import data
import model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils import batchify, get_batch, repackage_hidden
from adabound import AdaBound
from adabelief_pytorch import AdaBelief
from yogi import Yogi
from MSVAG import MSVAG
from RAdam import RAdam
from AdamW import AdamW
from fromage import Fromage

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
        
sys.stdout = Logger("/workspace1/ow120/DDAM/Graph_NN_case/cifar_10/Adabelief-Optimizer/PyTorch_Experiments/LSTM/lstm_1layer_3.txt")



parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 value')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='bets2 value')
parser.add_argument('--clip', type=float, default=None,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--num_gpu', default=4, type=int, help='GPU to use for simulated parallel training.')
parser.add_argument('--sigma_lr', default=0.08, type=float, help='sigma multiplier updating')
parser.add_argument('--rho_lr', default=10000, type=float, help='rho multiplier updating')
parser.add_argument('--beta_rmsprop', default=0.999, type=float, help='rmsprop coefficients beta_2')
parser.add_argument('--lr-gamma', nargs='+', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--eps', type=float,  default=1e-8)
parser.add_argument('--l2_lambda', default=0.00000, type=float, help='weight decay for global w')
parser.add_argument('--eps_sqrt', type=float,  default=1e-8)
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--run', type=int, default=0,
                    help='Number of runs')
args = parser.parse_args()

args.save = args.save + '-niter-{}'.format(args.epochs) + '-optimizer-{}'.format(args.optimizer) + '-nlayers{}'.format(args.nlayers) + \
            '-lr{}'.format(args.lr) + '-clip-{}'.format(args.clip) +'-eps{}'.format(args.eps) \
            +'-epsqrt{}'.format(args.eps_sqrt) + '-betas-{}-{}'.format(args.beta1, args.beta2) + '-run{}'.format(args.run) + '-wdecay{}-when-{}'.format(args.wdecay, args.when)

args.tied = True

if not os.path.exists('curve'):
    os.mkdir('curve')

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
print('learning rate decay is:', args.lr_gamma)
###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)

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
        #W_n_avg[i] = W_n_avg[i] + P_n_avg[i] / tau_lr[i]
        W_n_avg[i] = ((np.float64(tau_lr)*W_n_avg[i].double())/ (np.float64(tau_lr) + np.float64(l2_lambda))).float() + P_n_avg[i]/(tau_lr + l2_lambda)
        W_n_avg[i].detach()

    #del P_n_avg
    #gc.collect()
    return W_n_avg
    
def generate_W_global_2nd_order(num_batches, W_n_list, P_n_list, Z_n_list, tau_lr, alpha, epsilon = args.eps):
    W_n_avg = average_parameters(num_batches, W_n_list, alpha)
    P_n_avg = average_parameters(num_batches, P_n_list, alpha)
    Z_n_avg = average_parameters(num_batches, Z_n_list, alpha)
    for i in range(len(W_n_avg)):
        W_n_avg[i] = W_n_avg[i] + P_n_avg[i] / (tau_lr[i]*(torch.sqrt(Z_n_avg[i]) + epsilon))
        #W_n_avg[i] = W_n_avg[i] + P_n_avg[i] / tau_lr[i]
        W_n_avg[i].detach()

    #del P_n_avg
    #gc.collect()
    return W_n_avg
    
def get_sub_batch(source, i, gpu_idx, args, seq_len_sub=None, seq_len = None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i) 
    
    #seq_len_sub = min(seq_len_sub if seq_len_sub else args.bptt, (len(source) - 1 - i)//args.num_gpu)
    #seq_len_sub = min(seq_len_sub if seq_len_sub else args.bptt, (len(source) - 1 - i) - (args.num_gpu-1)*((len(source) - 1 - i)//args.num_gpu))
    
    if seq_len == len(source) - 1 - i:
    
        seq_len_sub = seq_len // args.num_gpu
        if gpu_idx == (args.num_gpu-1):
            data = source[i+gpu_idx*seq_len_sub:i+seq_len]
            target = source[i+1+gpu_idx*seq_len_sub:i+1+seq_len].view(-1)
        
        else:
            
            data = source[i+gpu_idx*seq_len_sub:i+gpu_idx*seq_len_sub+seq_len_sub]
            target = source[i+1+gpu_idx*seq_len_sub:i+1+gpu_idx*seq_len_sub+seq_len_sub].view(-1)
            
    else:
        
        if gpu_idx == (args.num_gpu-1):
            data = source[i+gpu_idx*seq_len_sub:i+seq_len]
            target = source[i+1+gpu_idx*seq_len_sub:i+1+seq_len].view(-1)
        
        else:
            data = source[i+gpu_idx*seq_len_sub:i+gpu_idx*seq_len_sub+seq_len_sub]
            target = source[i+1+gpu_idx*seq_len_sub:i+1+gpu_idx*seq_len_sub+seq_len_sub].view(-1)
    
    return data, target



# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

###############################################################################
# intial settings for mPiAM
###############################################################################
W_n_0 = [param.clone().detach().requires_grad_(True) for param in params]    
W_b_initial = [[param.clone() for param in W_n_0] for _ in range(args.num_gpu)]
P_b_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(args.num_gpu)]
#Z_b_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(args.num_gpu)]
accumulators_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(args.num_gpu)]
velocities_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(args.num_gpu)]


sigma_lr = args.sigma_lr
alpha_b = [1/args.num_gpu for _ in range(args.num_gpu)]
print('alpha_b is: ', alpha_b)
decay_count = 0

zero_grad(params)
#optimizer.zero_grad()
#learning_rate_current = 1/(args.sigma_lr + args.rho_lr)
learning_rate_current = [args.lr for _ in range(len(W_n_0))]
sigma_lr_current = [args.sigma_lr for _ in range(len(W_n_0))]
rho_lr_current = [1/args.lr - args.sigma_lr for _ in range(len(W_n_0))]
W_global = generate_W_global(args.num_gpu, W_b_initial, P_b_initial, sigma_lr_current, alpha_b, args.l2_lambda)
epsilon = 1e-7
updated_iteration = 1.0
epoch_loss = 0
#best_acc = args.baseline_acc

# At any point you can hit Ctrl + C to break out of training early.
try:
    #optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'fromage':
        optimizer = Fromage(params, lr=args.lr)
    if args.optimizer == 'adamw':
        optimizer = AdamW(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'radam':
        optimizer = RAdam(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer.lower() == 'adabelief':
        optimizer = AdaBelief(params, lr=args.lr, weight_decay=args.wdecay,
                             eps=args.eps, betas=(args.beta1, args.beta2))
    if args.optimizer == 'adabound':
        optimizer = AdaBound(params, lr=args.lr, weight_decay=args.wdecay, final_lr=30, gamma=1e-3)
    if args.optimizer == 'amsbound':
        optimizer = AdaBound(params, lr=args.lr, weight_decay=args.wdecay, final_lr=30, gamma=1e-3, amsbound=True)
    elif args.optimizer == 'yogi':
        optimizer =  Yogi(params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.wdecay)
    elif args.optimizer == 'msvag':
        optimizer = MSVAG(params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.wdecay)

    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        #train_loss = train()
        
        ############################## model train steps ##############################
        if args.model == 'QRNN': model.reset()
        total_loss = 0
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(args.batch_size)
        batch, i = 0, 0
        indices = np.random.permutation(train_data.size(0)-1-1)
        #sigma_lr_current = ((1/(args.sigma_lr + args.rho_lr))/learning_rate_current) * args.sigma_lr
        #sigma_lr_current = (args.lr/learning_rate_current) * args.sigma_lr
        print('Current LR is:', learning_rate_current)
        print('Current sigma_lr is:', sigma_lr_current)
        #print('train data size is:', train_data.size(0))
        #while i < train_data.size(0) - 1 - 1:
        while i < train_data.size(0) - 1 - args.num_gpu:
            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            if len(train_data) - 1 - i - seq_len < args.num_gpu and (len(train_data) - 1 - i) > seq_len:
                seq_len = len(train_data) - 1 - i # to avoid the final sampled sequence length shorter than num_gpu
            
            # There's a very small chance that it could select a very long sequence length resulting in OOM
            # seq_len = min(seq_len, args.bptt + 10)
    
            #lr2 = optimizer.param_groups[0]['lr']
            #optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
            lr2 = learning_rate_current[0]
            learning_rate_current[0] = lr2 * seq_len / args.bptt
            sigma_lr_current[0] = (args.lr/learning_rate_current[0]) * args.sigma_lr
            rho_lr_current[0] = (1/learning_rate_current[0]) - sigma_lr_current[0]
            #print('The learning rate is:', learning_rate_current[0])
            model.train()
            #data, targets = get_batch(train_data, i, args, seq_len=seq_len)
            alpha_b = []
            #print('################################')
            #print('The i is:', i)
            #print('The sequence length is:', seq_len)
            for sb in range(args.num_gpu):
                with torch.no_grad():  # Disable gradient tracking
                    for param, w in zip(params, W_global):
                        param.copy_(w)
                if sb == args.num_gpu - 1:
                    seq_len_sub = seq_len - (seq_len // args.num_gpu)*(args.num_gpu-1)
                else:
                    seq_len_sub = seq_len // args.num_gpu
    
                
                alpha_b.append(seq_len_sub/seq_len)
                #data_sub, target_sub = get_batch(train_data, i, args, seq_len=seq_len_sub)
                data_sub, target_sub = get_sub_batch(train_data, i, sb, args, seq_len_sub=int(alpha_b[0]*seq_len), seq_len=seq_len)
                #print('number of data_sub is:', len(data_sub))
                W_n = W_b_initial[sb]
                P_n = P_b_initial[sb]
                #Z_n = Z_b_initial[sb]
                accumulators = accumulators_initial[sb]
                #velocities = velocities_initial[sb]
                
                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                hidden = repackage_hidden(hidden)
                #optimizer.zero_grad()
                zero_grad(params)
        
                output, hidden, rnn_hs, dropped_rnn_hs = model(data_sub, hidden, return_h=True)
                #print('the sub-batch label size is:', target_sub.size())
                #print('the sub-batch output size is:', output.size())
                raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, target_sub)
                
        
                loss = raw_loss
                total_loss += raw_loss.data
                # Activiation Regularization
                if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
                # Temporal Activation Regularization (slowness)
                if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
                loss.backward()
        
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
                #optimizer.step()
                #gradients = [param.grad + args.wdecay*param for param in model.parameters()]
                gradients = [
                                param.grad + args.wdecay * param if param.grad is not None else None 
                                for param in params
                            ]
                
                '''for i, (param_wn, param_pn, gradient, param_wg, accumulator, velocity) in enumerate(zip(W_n, P_n, gradients, W_global, accumulators, velocities)):
                    print('Parameters shape is:', param_wn.shape)
                    if gradient is not None:
                        
                        print('Gradient shape is:', gradient.shape)'''
                
                with torch.no_grad():

                    for ii, (param_wn, param_pn, gradient, param_wg, accumulator, lr_current, sigma_lr_i, rho_lr_i) in enumerate(zip(W_n, P_n, gradients, W_global, accumulators, learning_rate_current, sigma_lr_current, rho_lr_current)):
                        if gradient is not None:
                            #velocity.mul_(args.beta1).add_((1 - args.beta1) * (gradient + param_pn))
                            accumulator.mul_(args.beta_rmsprop).add_((1 - args.beta_rmsprop) * (gradient + param_pn).pow(2))
                            #accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) * gradient.pow(2))
                            #accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) *  param_pn.pow(2))
                            
                            
                            bias_correction2 = 1 - args.beta_rmsprop** updated_iteration                        
                            corrected_accumulator = accumulator / (bias_correction2)
                            #bias_correction1 = 1 - args.beta1** updated_iteration                        
                            #corrected_velocity= velocity / (bias_correction1)
                            
                            
                            
                            
                            #delta = param_wg - lr_current * (gradient+ param_pn)/(torch.sqrt(corrected_accumulator) + args.eps)
                            delta = param_wg -  (gradient+ param_pn)/(sigma_lr_i + rho_lr_i * (torch.sqrt(corrected_accumulator) + args.eps))
                            
                            
                            param_wn.copy_(delta.detach())
                            param_pn.add_(sigma_lr_i * (param_wn - param_wg))
                            #param_zn.mul_(args.beta_rmsprop).add_((1 - args.beta_rmsprop) * (param_pn).pow(2))
                        
                #zero_grad(model.parameters())
                del loss
                del output
                
            updated_iteration += 1
            with torch.no_grad():
                #W_global = generate_W_global(num_gpu, W_b_initial, P_b_initial, sigma_lr, alpha_b_n[(update_count-num_gpu):update_count])
                W_global = generate_W_global(args.num_gpu, W_b_initial, P_b_initial, sigma_lr_current, alpha_b, args.l2_lambda)
                #W_global = generate_W_global_2nd_order(args.num_gpu, W_b_initial, P_b_initial, Z_b_initial, sigma_lr_current, alpha_b)
                for param, w in zip(params, W_global):
                    param.copy_(w)
                
                
            
    
                
            #optimizer.param_groups[0]['lr'] = lr2
            learning_rate_current[0] = lr2
    
            if batch % args.log_interval == 0 and batch > 0:
                #cur_loss = total_loss.item() / args.log_interval
                cur_loss = total_loss.item() / (args.log_interval * args.num_gpu)
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                    epoch, batch, len(train_data) // args.bptt, learning_rate_current[0],
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
                total_loss = 0
                start_time = time.time()
            ###
            batch += 1
            i += seq_len
        train_loss = math.exp(cur_loss)
        #############################################################################
        
        
        
        
        
         
        train_losses.append(train_loss) 
        if 't0' in optimizer.param_groups[0]:
            #tmp = {}
            #for prm in model.parameters():
            #    tmp[prm] = prm.data.clone()
            #    if 'ax' in optimizer.state[prm]:
            #        prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(test_data, test_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)

            if val_loss2 < stored_loss:
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss2

            #for prm in model.parameters():
            #    prm.data = tmp[prm].clone()

            val_losses.append(math.exp(val_loss2))
        else:
            val_loss = evaluate(test_data, test_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)


            best_val_loss.append(val_loss)

            val_losses.append(math.exp(val_loss))

        if epoch in args.when:
            print('Saving model before learning rate decreased')
            model_save('{}.e{}'.format(args.save, epoch))
            #print('Dividing learning rate by 10')
            if decay_count > len(args.lr_gamma):
                lr_decay = args.lr_gamma[-1]
            else:
                lr_decay = args.lr_gamma[decay_count]
            print('The decay rate is:', lr_decay)
            #if epoch == 150:
                #learning_rate_current =  [ii * 0.12 for ii in learning_rate_current]
            
            learning_rate_current =  [ii * lr_decay for ii in learning_rate_current]
            sigma_lr_current = [ (args.lr/learning_rate_current[jj]) * args.sigma_lr for jj in range(len(sigma_lr_current))] 
            rho_lr_current = [1/learning_rate_current[jj] - sigma_lr_current[jj] for jj in range(len(sigma_lr_current))]
            decay_count += 1
            #for param_group in optimizer.param_groups:
                #param_group['lr'] /= 10.
                

        #print(train_losses)
        torch.save({'train_loss': train_losses, 'test_loss': val_losses},
                   os.path.join('curve', args.save))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
