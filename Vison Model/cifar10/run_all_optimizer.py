import os

# 1-layer lstm
cmd = ' python main.py --optim adabelief --lr 1e-3 --eps 1e-8 --beta1 0.9 --beta2 0.999 --momentum 0.9'
os.system(cmd)


cmd = ' python main.py --optim adabound --lr 1e-3 --eps 1e-8 --beta1 0.9 --beta2 0.999 --momentum 0.9 --gamma 0.001'
os.system(cmd)

cmd = ' python main.py --optim adam --lr 1e-3 --eps 1e-8 --beta1 0.9 --beta2 0.999 --momentum 0.9'
os.system(cmd)

cmd = ' python main.py --optim adamw --lr 1e-3 --eps 1e-8 --beta1 0.9 --beta2 0.999 --momentum 0.9 --weight_decay 0.01'
os.system(cmd)

cmd = ' python main.py --optim fromage --lr 1e-2 --eps 1e-8 --beta1 0.9 --beta2 0.999 --momentum 0.9'
os.system(cmd)


cmd = ' python main.py --optim msvag --lr 1e-1 --eps 1e-8 --beta1 0.9 --beta2 0.999 --momentum 0.9'
os.system(cmd)

cmd = ' python main.py --optim radam --lr 1e-3 --eps 1e-8 --beta1 0.9 --beta2 0.999 --momentum 0.9'
os.system(cmd)

cmd = ' python main.py --optim sgd --lr 1e-1 --eps 1e-8 --momentum 0.9'
os.system(cmd)

cmd = ' python main.py --optim yogi --lr 1e-3 --eps 1e-3 --beta1 0.9 --beta2 0.999 --momentum 0.9'
os.system(cmd)