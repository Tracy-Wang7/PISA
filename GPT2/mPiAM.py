###########################################################################

# for this mPiAM, if we still keep baseline experiment as 4 batches/updating iterations per epoch
# and we still keep 4 batches, and split each batch into 4 mini-batches, we select one mini-batch from each batches, and do one time updating iteration

###########################################################################


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

num_gpu = 4
#num_minibatches = 4
num_updating_per_epoch = 4
accumulation_steps_per_batch = []
accumulation_steps_per_minibatch = []
accumulation_steps_per_update = []
for ii in range(num_gpu-1):
    accumulation_steps_per_batch.append(int(len(train_dataloader)/num_gpu))
    
accumulation_steps_per_batch.append(int(len(train_dataloader) - sum(accumulation_steps_per_batch)))


for ii in range(num_gpu):
    num_batch_size = accumulation_steps_per_batch[ii]
    for jj in range(num_updating_per_epoch-1):
        accumulation_steps_per_minibatch.append(int(num_batch_size/num_updating_per_epoch))
        
    accumulation_steps_per_minibatch.append(int(num_batch_size-(num_batch_size//num_updating_per_epoch)*(num_updating_per_epoch-1)))

for ii in range(num_updating_per_epoch):
    for jj in range(num_gpu):
        accumulation_steps_per_update.append(accumulation_steps_per_minibatch[ii+num_updating_per_epoch*jj])

print('The accumulation_steps_per_minibatch is: ', accumulation_steps_per_minibatch)
print('The accumulation_steps_per_update is: ', accumulation_steps_per_update)
alpha_b = [i/sum(accumulation_steps_per_batch) for i in accumulation_steps_per_batch]
alpha_b_n = []
for ii in range(len(accumulation_steps_per_update)):
    alpha_b_n.append(accumulation_steps_per_update[ii]/accumulation_steps_per_batch[ii%num_gpu])
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

# Initialize weights
W_n_0 = [param.clone().detach().requires_grad_(True) for param in model.parameters()]

W_n_set, P_n_set, Z_n_set = [], [], []

W_b_initial = [[param.clone() for param in W_n_0] for _ in range(num_gpu)]
P_b_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(num_gpu)]
accumulators_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(num_gpu)]

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

word_embeddings = model.transformer.wte.weight # Word Token Embeddings


epochs = 150
learning_rate = 2e-5
warmup_steps = 1e2
epsilon = 1e-8
# optim = Adam(model.parameters(), lr=5e-5)
optim = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))
    

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
            

total_steps = num_gpu * epochs

    
total_t0 = time.time()

training_stats = []
batch_count = 0  # Track the current batch number
update_count = 0  # Track how many updates have been performed
sigma_lr = 0.05 #0.08 #1000 suddenly increasiing # 9000 slow
rho_lr = 10000 #10000 #1000 # 1000
beta_rmsprop = 0.999
epsilon = 1e-8
rho = 0.9
beta1 = 0.9
beta2 = 0.999
#W_global = generate_W_global(num_gpu, W_b_initial, P_b_initial, sigma_lr, alpha_b)
W_global = [param.clone() for param in W_n_0]
current_step = 0


m = [torch.zeros_like(param, device=device) for param in model.parameters()]
v = [torch.zeros_like(param, device=device) for param in model.parameters()]
    

train_loss_record = []
current_step = 1
for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0
    update_count = 0
    

    model.train()  # `train` just changes the *mode* (train vs. eval), it doesn't *perform* the training.

    
    for step, batch in enumerate(train_dataloader):     # step from enumerate() = number of batches

        b_input_ids = batch[0].to(device)   # tokens (of multiple documents in a batch)
        b_labels    = batch[0].to(device)
        b_masks     = batch[1].to(device)   # mask of [1] for a real word, [0] for a pad

        #model.zero_grad()
        # loss = model(X.to(device), attention_mask=a.to(device), labels=X.to(device)).loss
        with torch.no_grad():  # Disable gradient tracking
            for param, w in zip(model.parameters(), W_global):
                param.copy_(w)
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
            
            #sigma_lr = get_linear_schedule_with_warmup_custom(current_step, warmup_steps, total_steps, sigma_lr)
            #rho_lr = get_linear_schedule_with_warmup_custom(current_step, warmup_steps, total_steps, rho_lr)
            learning_rate_current =  get_linear_schedule_with_warmup_custom(current_step, warmup_steps, total_steps, learning_rate)
            W_n = W_b_initial[update_count%num_gpu]
            P_n = P_b_initial[update_count%num_gpu]
            accumulators = accumulators_initial[update_count%num_gpu]
            gradients = [param.grad for param in model.parameters()]
            
            with torch.no_grad():

                for i, (param_wn, param_pn, gradient, param_wg, accumulator) in enumerate(zip(W_n, P_n, gradients, W_global, accumulators)):
                    accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) * (gradient+param_pn).pow(2)) 
                    #delta = 1 / (rho_lr + sigma_lr) * ((rho_lr+sigma_lr) * param_wg - (gradient + param_pn)/(torch.sqrt(accumulator) + epsilon))
                    delta = param_wg - learning_rate_current * (gradient + param_pn)/(torch.sqrt(accumulator) + epsilon)
                    param_wn.copy_(delta.detach())
                    param_pn.add_(sigma_lr * (param_wn - param_wg))
                    
                #manual_adam_update(model, gradients, m, v, learning_rate_current, beta1, beta2, epsilon)
                    
                
            model.zero_grad()
            batch_count = 0
            update_count += 1
            print(f"Epoch {epoch_i+1}, Update count {update_count}. Current_lr: {learning_rate_current}")
            
            del loss
            del outputs
            if update_count % num_gpu == 0:
                current_step += 1
                print(f"This is {update_count//num_gpu} updating iteration in {epoch_i} epoch")
            
                with torch.no_grad():
                    #W_global = generate_W_global(num_gpu, W_b_initial, P_b_initial, sigma_lr, alpha_b_n[(update_count-num_gpu):update_count])
                    W_global = generate_W_global(num_gpu, W_b_initial, P_b_initial, sigma_lr, alpha_b)
                    for param, w in zip(model.parameters(), W_global):
                        param.copy_(w)

    #with torch.no_grad():
        #W_global = generate_W_global(num_gpu, W_b_initial, P_b_initial, sigma_lr, alpha_b)
    # Calculate the average loss over all the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    train_loss_record.append(avg_train_loss)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))


    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        with torch.no_grad():

            outputs  = model(input_ids = b_input_ids,
                             attention_mask = b_masks,
                             labels = b_labels)

            loss = outputs[0]

        batch_loss = loss.item()
        total_eval_loss += batch_loss

    avg_val_loss = total_eval_loss / len(validation_dataloader)

    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))


    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
print("")
print("Training complete!")


print("Saving model to %s" % model_save_path)
np.save(f"mPiAM_loss.npy", np.array(train_loss_record))
# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
# model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
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
print('#########################################################################')
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