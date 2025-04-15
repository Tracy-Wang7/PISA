To reproduce results in Figure 6 and table 5, run the following command, each experiment we run five times and average results:

```bash
python main_SPiAM.py  --Train --dataset cifar10 --sigma_lr_generator 0.1 --sigma_lr_discriminator 0.1 --num_gpu 4 --optimizer rmsprop --lr 1e-3 --n_critic 1
```
