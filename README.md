# Preconditioned Inexact Stochastic ADMM for Deep Models
---
This is the code of paper [Preconditioned Inexact Stochastic ADMM for Deep Models](https://arxiv.org/abs/2502.10784)

## Requirements
Our main experiments and analysis are conducted on the following environment:
- CUDA (12.7)
- cuDNN (9.1.0)
- Pytorch (2.4.0)
- Transformers (4.45.1)
- NumPy (1.24.3)

To run the current records for Vision Model, WGAN, LSTM and Data Heterogeneity, run the following commands for the set up:

```bash
conda env create -f vision_tasks.yml
```
The set up for other tasks will be sepcified in their corresponding folders.

All LLM experiments are conducted on four NVIDIA H100-80GB GPUs, while all other experiments run on a single such GPU.

---
## Sections in the paper

✅ **Data Heterogenerity** corresponds to Section **Data Heterogeneity**

✅ **Vision Model** corresponds to Section **Classification by VMs**

✅ **GPT2** corresponds to Appendix **Training LLMs**

✅ **GPT2_new_version** corresponds to Section **Training LLMs**

✅ **RL** corresponds to Appendix **Games reward by RLMs**

✅ **WGAN** corresponds to Section **Generation tasks by GANs**

✅ **LSTM** corresponds to Appendix **Text prediction with RNNs**
