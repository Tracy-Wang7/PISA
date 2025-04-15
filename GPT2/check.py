import pandas as pd

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
from transformers import get_linear_schedule_with_warmup

from tqdm.auto import tqdm
import random
import datetime
import time

import nltk
import numpy as np
import bert_score
import os
import itertools
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"device {device}")

model_name = "gpt2-medium"  # options: ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
model_save_path = './model_more_epoches'

'''configuration = GPT2Config.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)

tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

input_sequence = "beef, salt, pepper"
input_ids = tokenizer.encode(input_sequence, return_tensors='pt')

model = model.to(device)'''
#combine both sampling techniques
'''sample_outputs = model.generate(
                              input_ids.to(device),
                              do_sample = True,
                              max_length = 120,
                              top_k = 50,
                              top_p = 0.85,
                              num_return_sequences = 3
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}...".format(i, tokenizer.decode(sample_output, skip_special_tokens = True)))
    print('  ---')'''
    
df_recipes = pd.read_csv('recipes_1000.csv')
df_recipes.reset_index(drop=True, inplace=True)

# df_recipes = df_recipes.iloc[:600]
print(list(df_recipes.columns))
print(f"data shape {df_recipes.shape}")



tokenizer = GPT2TokenizerFast.from_pretrained(model_name,
                                              bos_token='<|startoftext|>',
                                              eos_token='<|endoftext|>',
                                              unk_token='<|unknown|>',
                                              pad_token='<|pad|>'
                                             )


doc_lengths = []

for rec in df_recipes.itertuples():

    # get rough token count distribution
    tokens = nltk.word_tokenize(rec.ingredients + ' ' + rec.instructions)

    doc_lengths.append(len(tokens))

doc_lengths = np.array(doc_lengths)
# the max token length
print(f"% documents > 180 tokens: {round(len(doc_lengths[doc_lengths > 180]) / len(doc_lengths) * 100, 1)}%")
print(f"Average document length: {int(np.average(doc_lengths))}")


def form_string(ingredient,instruction):
    # s = f"<|startoftext|>Ingredients:\n{ingredient.strip()}\n\nInstructions:\n{instruction.strip()}<|endoftext|>"
    s = f"<|startoftext|>Ingredients: {ingredient.strip()}. " \
        f"Instructions: {instruction.strip()}<|endoftext|>"
    return s

def extract_string(recipe):
    str = recipe.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
    inst_pos = str.find('Instructions: ')
    ingredients = str[len('Ingredients: '): inst_pos-1]
    instructions = str[inst_pos+len('Instructions: '):]
    return ingredients, instructions

data = df_recipes.apply(lambda x:form_string(
    x['ingredients'], x['instructions']), axis=1).to_list()



# GPT2 is a large model. Increasing the batch size above 2 has lead to out of memory problems.
batch_size = 16
max_length = 180  # maximum sentence length

def average_parameters(num_train_env, list_vars, list_alpha):
    sum_vars = [torch.zeros_like(var) for var in list_vars[0]]
    for i in range(num_train_env):
        W_n = list_vars[i]
        alpha = list_alpha[i]
        sum_vars = [sum_ + alpha*update for sum_, update in zip(sum_vars, W_n)]
    return sum_vars


def generate_W_global(num_batches, W_n_list, P_n_list, tau_lr, alpha):
    W_n_avg = average_parameters(num_batches, W_n_list, alpha)
    P_n_avg = average_parameters(num_batches, P_n_list, alpha)
    for i in range(len(W_n_avg)):
        W_n_avg[i] = W_n_avg[i] + P_n_avg[i] / tau_lr
        W_n_avg[i].detach()

    #del P_n_avg
    #gc.collect()
    return W_n_avg
    
    
'''def get_linear_schedule_with_warmup_custom(current_step, warmup_steps, total_steps, base_lr):
    
    if current_step < warmup_steps:
        # During warmup phase, increase learning rate linearly
        #lr = base_lr * (current_step / warmup_steps)
        lr = base_lr * (warmup_steps / current_step) ## be careful for current step should begin at 1, dont begin at 0 
    else:
        # After warmup, decrease learning rate linearly
        #lr = base_lr * (1 - (current_step - warmup_steps) / (total_steps - warmup_steps))
        lr = base_lr * ( ((total_steps+1) - warmup_steps) / ((total_steps+1) - current_step
    return lr'''
    
def get_linear_schedule_with_warmup_custom(current_step, warmup_steps, total_steps, base_lr):
    """
    Custom function to linearly modify the learning rate with warmup.

    Args:
        current_step (int): Current step in training.
        warmup_steps (int): Number of steps for learning rate warmup.
        total_steps (int): Total number of training steps.
        base_lr (float): The initial base learning rate.

    Returns:
        float: The modified learning rate.
    """
    
    if current_step < warmup_steps:
        # During warmup phase, increase learning rate linearly
        lr = base_lr * (current_step / warmup_steps)
    else:
        # After warmup, decrease learning rate linearly
        lr = base_lr * (1 - (current_step - warmup_steps) / (total_steps - warmup_steps))

    return lr



def split_data_loader(data_loader, num_subsets, split_indices):
    # Get the number of batches in the data_loader
    total_batches = len(data_loader)
    #subset_size = total_batches // num_subsets  # Batches per subset
    
    # Turn data_loader into an iterable
    data_iter = iter(data_loader)
    
    # Split the iterable into sub-dataloaders
    sub_data_loaders = []
    for subset_size in split_indices:
        # Extract a subset of batches
        subset_batches = list(itertools.islice(data_iter, subset_size))
        
        # Create a DataLoader for each subset of batches
        sub_data_loaders.append(subset_batches)
    
    return sub_data_loaders


# standard PyTorch approach of loading data in using a Dataset class.
class RecipeDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.input_ids = []
        self.attn_masks = []
        self.origin_ingredients = []
        self.origin_instructions = []

        for recipe in data:
            encodings = tokenizer.encode_plus(recipe,
                                              truncation=True,
                                              padding='max_length',
                                              max_length=max_length,
                                              return_tensors='pt'       # return PyTorch tensor
                                             )
            self.input_ids.append(torch.squeeze(encodings['input_ids'],0))
            # attention_mask tells model not to incorporate these PAD tokens into its interpretation of the sentence
            self.attn_masks.append(torch.squeeze(encodings['attention_mask'],0))
            ingredients, instructions = extract_string(recipe)
            self.origin_ingredients.append(ingredients)
            self.origin_instructions.append(instructions)


    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.input_ids[idx], self.attn_masks[idx], self.origin_ingredients[idx], self.origin_instructions[idx]
        


def infer(prompt, model):
    input = f"<|startoftext|>Ingredients: {prompt.strip()}"
    input = tokenizer(input, return_tensors="pt")
    input_ids      = input["input_ids"]
    attention_mask = input["attention_mask"]

    output = model.generate(input_ids.to(device),
                            attention_mask=attention_mask.to(device),
                            max_new_tokens=max_length,
                            # temperature = 0.5,
                            do_sample = True, top_k = 50, top_p = 0.85)
                            # num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output
        
dataset = RecipeDataset(data, tokenizer)

# Split into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

# Create the DataLoaders for our training and validation datasets.
# We'll take training samples in random order.
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# Calculate the length of the dataset
total_samples = len(train_dataloader)


indices = np.arange(total_samples)

# Split the indices into 4 parts
split_indices = np.array_split(indices, 4)

# Create a list to store the new DataLoaders
for ii in range(len(split_indices)):
    print(len(split_indices[ii]))

num_subsets = 4
accumulation_steps_per_update = []
for ii in range(num_subsets-1):
    accumulation_steps_per_update.append(int(len(train_dataloader)/num_subsets))
    
accumulation_steps_per_update.append(int(len(train_dataloader) - sum(accumulation_steps_per_update)))
print('The accumulation_steps_per_update is: ', accumulation_steps_per_update)    

sub_data_loaders = split_data_loader(train_dataloader, num_subsets, accumulation_steps_per_update)
for i, sub_loader in enumerate(sub_data_loaders):
    print(f"Sub-loader {i+1} contains {len(sub_loader)} batches")