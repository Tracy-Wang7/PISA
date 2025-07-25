To reproduce results in Table 3, run the following command. Each network we run 5 times and average results.  
VGG:
```bash
python l2_lower_loss_mPiAM_varying_rho_sigma.py --model vgg --optim radam --eps 1e-8 --sigma_lr 4.5 --rho_lr 3e4 --beta1 0.9 --beta2 0.999 --momentum 0.9 --batchsize 128  --total_epoch 205 --decay_epoch 10  --lr-gamma 0.9 --baseline_acc 0.9103 --beta_rmsprop 0.995 --weight_decay 2.5e-4 --l2_lambda 4e-4
```
Resnet:
```bash
python lower_loss_mPiAM_training_procedure.py --model resnet --optim adamw --eps 1e-8 --sigma_lr 0.1 --rho_lr 5e3 --beta1 0.9 --beta2 0.999 --momentum 0.9 --batchsize 128  --total_epoch 205 --decay_epoch 3 --lr-gamma 0.85 --weight_decay 5e-5 --baseline_acc 95.00 --beta_rmsprop 0.999
```
Densenet:
```bash
python lambda_densenet.py --model densenet --baseline_acc 94.81 --optim adam --eps 1e-8 --sigma_lr 0.8 --rho_lr 2e3 --beta1 0.9 --beta2 0.999 --momentum 0.9 --batchsize 256  --total_epoch 201 --decay_epoch 50  --weight_decay 5e-4 --lr-gamma 0.25 --beta_rmsprop 0.999 --l2_lambda 5e-5
```
