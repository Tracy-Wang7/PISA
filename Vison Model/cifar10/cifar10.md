To reproduce results in Table 2, run the following command. Each network we run 5 times and average results.  
VGG:
```bash
python lower_loss_mPiAM_varying_rho_sigma.py --model vgg --optim adamw --eps 1e-8 --sigma_lr 6.5 --rho_lr 5e4 --beta1 0.9 --beta2 0.999 --momentum 0.9 --batchsize 128  --total_epoch 205 --decay_epoch 10  --lr-gamma 0.9 --baseline_acc 0.9103 --beta_rmsprop 0.999 --weight_decay 2.5e-4
'''
Resnet:
```bash
python lower_loss_mPiAM_training_procedure.py --model resnet --optim adamw --eps 1e-8 --sigma_lr 0.1 --rho_lr 5e3 --beta1 0.9 --beta2 0.999 --momentum 0.9 --batchsize 128  --total_epoch 205 --decay_epoch 3 --lr-gamma 0.85 --weight_decay 5e-5 --baseline_acc 95.00 --beta_rmsprop 0.999
'''
