# Train GPT2 with SISA
- ⚡ 
The updated version of SISA now support parallel computing, and allow efficient communication between devices

To run the current records for all GPT2 experiments in Section 5.4, run the following commands for the set up:

```bash
conda env create -f gpt2.yml
```
Then directly run the script in each directory

# Baseline Optimizers
You can find each baseline optimizers in Figure 4 from the link below

SOAP: (https://github.com/nikhilvyas/SOAP/tree/bbce86e890d3b697380f4376acb600c2d6c3d203)

Shampoo: (https://github.com/facebookresearch/optimizers/tree/ad2809a291c01859f68fcabbcb49a2aa75fd7827/distributed_shampoo)

Muon: (https://github.com/KellerJordan/modded-nanogpt/tree/master)

Adam-mini: (https://github.com/zyushun/Adam-mini)

