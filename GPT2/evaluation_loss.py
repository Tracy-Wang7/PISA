import pandas as pd

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler, Subset

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
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"device {device}")

model_name = "gpt2-medium"  # options: ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
#model_save_path = "/workspace1/ow120/DDAM/gpt2-pytorch/gpt_finetuning/01-Recipes-Generation-GPT2/SPiAM/AdamW/model_epoch_14/"
model_save_path = "/workspace1/ow120/DDAM/gpt2-pytorch/gpt_finetuning/01-Recipes-Generation-GPT2/SPiAM/SPiAM/model_epoch_14/"

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
batch_size = 64
max_length = 180  # maximum sentence length

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

#train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
split_indices = torch.load("dataset_splits.pth")
train_indices = split_indices["train_indices"]
val_indices = split_indices["val_indices"]

# Recreate the train and validation datasets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

# Create the DataLoaders for our training and validation datasets.
# We'll take training samples in random order.
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

##################### Validation before fine-tuning ##################### 
'''configuration = GPT2Config.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)
model.to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
scores_before = []
#for i in range(len(validation_dataloader)):
for i in range(10):
    ingredients = val_dataset[i][2]
    reference = val_dataset[i][3]
    candidate = infer(ingredients, model)
    P, R, F1 = bert_score.score([reference], [candidate], lang="en", device=device)
    scores_before.append(F1.mean().item())
    
print('#########################################################################')
print('The semantic similarity before fine-tuning is:', sum(scores_before)/len(scores_before))
print('#########################################################################')'''
######################################################################### 

for idx in [1,4,14, 24, 29, 44]:

    #model_save_path = f"/workspace1/ow120/DDAM/gpt2-pytorch/gpt_finetuning/01-Recipes-Generation-GPT2/SPiAM/SPiAM/model_epoch_{idx}/"
    #model_save_path = f"/workspace1/ow120/DDAM/gpt2-pytorch/gpt_finetuning/01-Recipes-Generation-GPT2/SPiAM_GPT_Small/AdamW/model_epoch_{idx}/"
    model_save_path = f"/workspace1/ow120/DDAM/gpt2-pytorch/gpt_finetuning/01-Recipes-Generation-GPT2/SPiAM_GPT_Small/SPiAM/model_epoch_{idx}/"

    model = GPT2LMHeadModel.from_pretrained(model_save_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_save_path)
    model.to(device)
    
    ##################### Validation after fine-tuning ##################### 
    validation_loss = 0
    for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_labels    = batch[0].to(device)
            b_masks     = batch[1].to(device)
    
            with torch.no_grad():
                outputs  = model(input_ids = b_input_ids, labels = b_labels, attention_mask = b_masks)
                loss = outputs[0]
                validation_loss+=loss.item()
        
    print('#########################################################################')
    print('The validation loss after fine-tuning is:', validation_loss/len(validation_dataloader))
    print('#########################################################################')
    ######################################################################### 
