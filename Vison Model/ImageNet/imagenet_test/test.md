You need to download ImageNet dataset first, and change the 'data' augment in new_mPiAM_l2_lambda.py to your own path.
To reproduce the results in table 3, run the following command.
```bash
python new_mPiAM_l2_lambda.py  --optimizer rmsprop --lr 5e-4 --sigma 1 --wd 1e-2 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decouple True --batch-size 256 --arch resnet18 --when 65 75 83 --gpu 0 --decay_rate 0.15 --l2_lambda 5e-6
```
