import pandas as pd

import torch
from torch.optim import AdamW, Adam, RMSprop
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

import sys
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class Logger(object):
    def __init__(self, fileN="record.txt"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()                 # flush the file after each write
    def flush(self):
        self.log.flush()
        
sys.stdout = Logger("/workspace1/ow120/DDAM/gpt2-pytorch/gpt_finetuning/01-Recipes-Generation-GPT2/baseline_adamw_small.txt")

device = "cuda:3" if torch.cuda.is_available() else "cpu"
print(f"device {device}")

#model_name = "gpt2-medium"  # options: ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
model_name = "gpt2"  # options: ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
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
def manual_adam_update(model, grads, m, v, lr, beta1, beta2, epsilon):
    with torch.no_grad():
        for param, grad, m_t, v_t in zip(model.parameters(), grads, m, v):
            if grad is None:
                continue

            # Update biased first moment estimate
            m_t[:] = beta1 * m_t + (1 - beta1) * grad

            # Update biased second moment estimate
            v_t[:] = beta2 * v_t + (1 - beta2) * grad ** 2

            # Compute bias-corrected first moment estimate
            m_hat = m_t 

            # Compute bias-corrected second moment estimate
            v_hat = v_t

            # Update parameters
            param -= lr * m_hat / (torch.sqrt(v_hat) + epsilon)
        
dataset = RecipeDataset(data, tokenizer)

# Split into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

split_indices = torch.load("dataset_splits.pth")
train_indices = split_indices["train_indices"]
val_indices = split_indices["val_indices"]

# Recreate the train and validation datasets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

'''train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
split_indices = {
    "train_indices": train_dataset.indices,
    "val_indices": val_dataset.indices,
}
torch.save(split_indices, "dataset_splits.pth")
print("Split indices saved!")'''

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

# Create the DataLoaders for our training and validation datasets.
# We'll take training samples in random order.
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

num_gpu = 4
accumulation_steps_per_update = []
for ii in range(num_gpu-1):
    accumulation_steps_per_update.append(int(len(train_dataloader)/num_gpu))
    
accumulation_steps_per_update.append(int(len(train_dataloader) - sum(accumulation_steps_per_update)))
print('The accumulation_steps_per_update is: ', accumulation_steps_per_update)

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


tokenizer = GPT2TokenizerFast.from_pretrained(model_name,
                                              bos_token='<|startoftext|>',
                                              eos_token='<|endoftext|>',
                                              unk_token='<|unknown|>',
                                              pad_token='<|pad|>'
                                             )

configuration = GPT2Config.from_pretrained(model_name, output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)
model = model.to(device)
# this step is necessary because I've added some tokens (bos_token, etc.) to the embeddings
# otherwise the tokenizer and model tensors won't match up
model.resize_token_embeddings(len(tokenizer))
print(f"Number of tokens: {len(tokenizer)}")

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

word_embeddings = model.transformer.wte.weight # Word Token Embeddings


#epochs = 150
epochs = 80
#learning_rate = 2e-5
#learning_rate = 3e-3 ### medium
learning_rate = 3e-3 ### small
warmup_steps = 1e2
epsilon = 1e-8
# optim = Adam(model.parameters(), lr=5e-5)
optim = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)
#optim = Adam(model.parameters(), lr = learning_rate, eps = epsilon)
#optim = RMSprop(model.parameters(), lr = learning_rate, eps = epsilon)
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))
    
# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
#total_steps = len(train_dataloader) * epochs
total_steps = num_gpu * epochs
# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optim,
                                            num_warmup_steps = warmup_steps,
                                            num_training_steps = total_steps)
    
total_t0 = time.time()

training_stats = []
batch_count = 0  # Track the current batch number
update_count = 0  # Track how many updates have been performed
train_loss_record = []
semantic_test_record = []

m = [torch.zeros_like(param, device=device) for param in model.parameters()]
v = [torch.zeros_like(param, device=device) for param in model.parameters()]
beta1 = 0.9
beta2 = 0.999
current_step = 0
t0 = time.time()
print('The length of trainloader is:', len(train_dataloader))
testing_epochs = [2, 5, 15, 25, 30, 45]
for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    #t0 = time.time()

    total_train_loss = 0
    update_count = 0
    model.zero_grad()

    model.train()  # `train` just changes the *mode* (train vs. eval), it doesn't *perform* the training.

    for step, batch in enumerate(train_dataloader):     # step from enumerate() = number of batches

        b_input_ids = batch[0].to(device)   # tokens (of multiple documents in a batch)
        b_labels    = batch[0].to(device)
        b_masks     = batch[1].to(device)   # mask of [1] for a real word, [0] for a pad

        #model.zero_grad()
        # loss = model(X.to(device), attention_mask=a.to(device), labels=X.to(device)).loss
        outputs = model(  input_ids = b_input_ids,
                          labels = b_labels,
                          attention_mask = b_masks,
                          token_type_ids = None
                        )

        loss = outputs[0]

        batch_loss = loss.item()
        total_train_loss += batch_loss
        current_accumulation_steps = accumulation_steps_per_update[update_count]
        

        # Get sample every x batches.
        '''if step % 10 == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

            #model.eval()

            #sample_output = infer("eggs, flour, butter, sugar", model)
            #print(sample_output)

            # `train` just changes the *mode* (train vs. eval), it doesn't *perform* the training.
            model.train()'''

        loss = loss / current_accumulation_steps
        loss.backward()
        batch_count += 1
        if batch_count == current_accumulation_steps:
            optim.step()
            scheduler.step()
            '''current_step +=1
            learning_rate_current =  get_linear_schedule_with_warmup_custom(current_step, warmup_steps, total_steps, learning_rate)
            gradients = [param.grad for param in model.parameters()]
            with torch.no_grad():                    
                manual_adam_update(model, gradients, m, v, learning_rate_current, beta1, beta2, epsilon)
            model.zero_grad()
            batch_count = 0
            update_count += 1
            current_lr = optim.param_groups[0]['lr']
            #print(f"Epoch {epoch_i+1}, Update count {update_count}. Current_lr: {current_lr}")
            print(f"Epoch {epoch_i+1}, Update count {update_count}. Current_lr: {learning_rate_current}")'''
            
            model.zero_grad()
            batch_count = 0
            update_count += 1
            current_lr = optim.param_groups[0]['lr']
            print(f"Epoch {epoch_i+1}, Update count {update_count}. Current_lr: {current_lr}")

    # Calculate the average loss over all the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    train_loss_record.append(avg_train_loss)

    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    
        
    ##################### Validation after fine-tuning ##################### 
    if (epoch_i+1) in  testing_epochs:
        '''model.eval()
        scores_after = []
        #for i in range(len(validation_dataloader)):
        for i in range(10):
            ingredients = val_dataset[i][2]
            reference = val_dataset[i][3]
            candidate = infer(ingredients, model)
            P, R, F1 = bert_score.score([reference], [candidate], lang="en", device=device)
            scores_after.append(F1.mean().item())
            
        print('#########################################################################')
        print('The semantic similarity after fine-tuning is:', sum(scores_after)/len(scores_after))
        print('#########################################################################')
        semantic_test_record.append(sum(scores_after)/len(scores_after))'''
        
        #save_path = f"SPiAM_GPT_Medium/AdamW/model_epoch_{epoch_i}"
        save_path = f"SPiAM_GPT_Small/AdamW/model_epoch_{epoch_i}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print('################################')
        print('Finsh saving models')

        
        
    #########################################################################
    
    if (epoch_i+1) == 45:
    
        np.save("SPiAM_GPT_Small/avg_loss_baseline_AdamW.npy", np.array(train_loss_record))
        #np.save("SPiAM/semantic_test_AdamW.txt", np.array(semantic_test_record))
        break
    
    
    
    # Measure how long this epoch took.
    #if epoch_i == 149:
    
    

training_time = format_time(time.time() - t0)

print("")
print("  Average training loss: {0:.2f}".format(avg_train_loss))
print("  Training epoch took: {:}".format(training_time))
#print(f"  Training epoch took {training_time:.2f} seconds")
np.save("SPiAM_GPT_Small/avg_loss_baseline_AdamW.npy", np.array(train_loss_record))
#np.save("SPiAM_GPT_Small/semantic_test_AdamW.txt", np.array(semantic_test_record))



print("")
print("Running Validation...")

print("")
print("Training complete!")


print("Saving model to %s" % model_save_path)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
# model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
'''np.save(f"loss_baseline_epoch.npy", np.array(train_loss_record))
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

model = GPT2LMHeadModel.from_pretrained(model_save_path)
tokenizer = GPT2TokenizerFast.from_pretrained(model_save_path)
model.to(device)

##################### Validation after fine-tuning ##################### 
scores_after = []
#for i in range(len(validation_dataloader)):
for i in range(10):
    ingredients = val_dataset[i][2]
    reference = val_dataset[i][3]
    candidate = infer(ingredients, model)
    P, R, F1 = bert_score.score([reference], [candidate], lang="en", device=device)
    scores_after.append(F1.mean().item())
    
print('#########################################################################')
print('The semantic similarity after fine-tuning is:', sum(scores_after)/len(scores_after))
print('#########################################################################')'''
######################################################################### 

# Using BLEU score to compare the real sentences with the generated ones
'''import statistics
from nltk.translate.bleu_score import sentence_bleu

scores=[]

for i in range(10):
    ingredients = val_dataset[i][2]
    reference = val_dataset[i][3]
    candidate = infer(ingredients)
    scores.append(sentence_bleu(reference, candidate))

print(statistics.mean(scores))
'''