All results in Figure 5 are produced based on Tianshou (https://github.com/thu-ml/tianshou) and mujoco (https://github.com/thu-ml/tianshou/tree/master/examples/mujoco).

Please first following the installation guidance of Tianshou and mujoco.

Then put a2c_SPiAM.py and ppo_SPiAM.py under the directory:
```bash
tianshou/tianshou/policy/modelfree/
```
And put mujoco_a2c_SPiAM.py and mujoco_ppo_SPiAM.py under the directory
```bash
tianshou/examples/mujoco/
```
To reproduce results in Figure 5 run the following command:

ANT:
```bash
python mujoco_ppo_SPiAM.py --task Ant-v3 --lr 5e-5
```

Humanoid:
```bash
python mujoco_ppo_SPiAM.py --task Humanoid-v3 --lr 1e-5 --sigma_lr 0.1
```

HalfCheetah:
```bash
python mujoco_a2c_SPiAM.py --task HalfCheetah-v3 --lr 5e-3 --sigma_lr 0.5
```

InvertedDoublePendulum:
```bash
python mujoco_a2c_SPiAM.py --task InvertedDoublePendulum-v2 --lr 5e-3 --sigma_lr 0.5
```
