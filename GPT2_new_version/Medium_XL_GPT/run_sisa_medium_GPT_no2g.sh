CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --standalone --nproc_per_node=4 --master_port 12345  train_gpt_sisa_lower_no_2ndgradient.py config/train_gpt2_medium_sisa.py
