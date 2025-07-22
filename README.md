# SISA
To run the current records for Vision Model, WGAN, LSTM and Data Heterogeneity, run the following commands for the set up:

```bash
conda env create -f vision_tasks.yml
```
The set up for other tasks will be sepcified in their corresponding folders.

All LLM experiments are conducted on four NVIDIA H100-80GB GPUs, while all other experiments run on a single such GPU.

---
## Sections in the paper

✅**Data Heterogenerity** corresponds to Section 5.2 **Data Heterogeneity**

✅**Vision Model** corresponds to Section 5.3 **Classification by VMs**

✅**GPT2** corresponds to Section 5.4 **Classification by VMs**

✅**GPT2_new_version** corresponds to Section 5.5 **Classification by VMs**

✅**RL** corresponds to Section 5.6 **Games reward by RLMs**

✅**WGAN** corresponds to Section 5.6 **Generation tasks by GANs**

✅**LSTM** corresponds to Section 5.6 **Text prediction with RNNs**
