To reproduce the results in Table 6, run the following command:

Layer1:
```bash
python main_mPiAM.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 201 --save PTB.pt --when 90 120 160 195 200 --clip 0.25 --beta1 0.9 --beta2 0.999 --optimizer adam --lr 2.5e-3  --eps 1e-12 --eps_sqrt 0.0 --nlayer 1 --run 0 --lr-gamma 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 --sigma_lr 0.04
```
Layer2:
```bash
python main_mPiAM.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 201 --save PTB.pt --when 140 165 190 198 --clip 0.25 --beta1 0.9 --beta2 0.999 --optimizer adam --lr 1.5e-2  --eps 1e-12 --eps_sqrt 0.0 --nlayer 2 --run 0 --lr-gamma 0.15 0.15 0.15 0.15 0.5 0.5 0.5 --sigma_lr 0.01
```

Layer3:
```bash
python main_mPiAM.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 201 --save PTB.pt --when 140 165 190 198 --clip 0.25 --beta1 0.9 --beta2 0.999 --optimizer adam --lr 1.5e-2  --eps 1e-12 --eps_sqrt 0.0 --nlayer 3 --run 0 --lr-gamma 0.15 0.15 0.15 0.15 0.5 0.5 0.5 --sigma_lr 0.01
```
