To reproduce results in Figure 6 and table 5, run the following command, each experiment we run five times and average results:

```bash
python main_SPiAM.py  --Train --dataset cifar10 --sigma_lr_generator 5e-5 --sigma_lr_discriminator 5e-5 --lr 5e-3 --num_gpu 2 --l2_lambda 1e-4 --n_critic 1
```

