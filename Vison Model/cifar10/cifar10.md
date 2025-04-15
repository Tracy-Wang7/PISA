To reproduce results in Table 2, run the following command. Each network we run 5 times and average results.  
VGG:
'''bash vgg cifar10 lower training loss:  python lower_loss_mPiAM_varying_rho_sigma.py --model vgg --optim adamw --eps 1e-8 --sigma_lr 6.5 --rho_lr 5e4 --beta1 0.9 --beta2 0.999 --momentum 0.9 --batchsize 128  --total_epoch 205 --decay_epoch 10  --lr-gamma 0.9 --baseline_acc 0.9103 --beta_rmsprop 0.999 --weight_decay 2.5e-4
