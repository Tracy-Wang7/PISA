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
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
import torch.autograd.profiler as profiler
import time
import itertools
os.environ["TOKENIZERS_PARALLELISM"] = "false"

'''model_name = "gpt2-medium"  # options: ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
model_save_path = './model_more_epoches'
    
df_recipes = pd.read_csv('recipes_1000.csv')
df_recipes.reset_index(drop=True, inplace=True)

# df_recipes = df_recipes.iloc[:600]
print(list(df_recipes.columns))
print(f"data shape {df_recipes.shape}")'''




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
    def __init__(self, data, tokenizer, max_length):
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
        


def infer(prompt, model, max_length):
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
    
    
    
    


def train(rank, world_size, sub_data_loaders, validation_dataloader, sigma_lr, rho_lr, beta_rmsprop, alpha_b, epsilon=1e-8, learning_rate = 2e-5, warmup_steps = 1e2, epoches = 500):
    try:
        # Initialize the process group
        os.environ['MASTER_ADDR'] = 'localhost'  # or the address of the master node if not localhost
        os.environ['MASTER_PORT'] = '12350'  # an open port on the master node
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{rank}")
        train_dataloader = sub_data_loaders[rank]
        #train_dataloader = sub_data_loaders

    
        # Set up the models and optimizers
        
        
        
        
        # Dummy data
        
        model_name = "gpt2-medium"  # options: ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name,
                                              clean_up_tokenization_spaces=True,
                                              bos_token='<|startoftext|>',
                                              eos_token='<|endoftext|>',
                                              unk_token='<|unknown|>',
                                              pad_token='<|pad|>'
                                             )

        configuration = GPT2Config.from_pretrained(model_name, output_hidden_states=False)
        model_save_path = './model_more_epoches'
        model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)
        model = model.to(device)
        # this step is necessary because I've added some tokens (bos_token, etc.) to the embeddings
        # otherwise the tokenizer and model tensors won't match up
        model.resize_token_embeddings(len(tokenizer))
        
        optimizer = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)
        total_steps = epoches * world_size
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = warmup_steps,
                                                    num_training_steps = total_steps)
        
        W_n = [param.clone().to(device) for param in model.parameters()]
        P_n = [torch.zeros_like(param).to(device) for param in model.parameters()]
        #Z_n = [torch.zeros_like(param).to(device) for param in model1.parameters()]
        accumulators = [torch.zeros_like(param) for param in model.parameters()]
        
        
        # calculate first then aggregation
        '''gathered_w_n = [[torch.zeros_like(param) for param in W_n] for _ in range(world_size)]
        gathered_p_n = [[torch.zeros_like(param) for param in P_n] for _ in range(world_size)]
        gathered_z_n = [[torch.zeros_like(param) for param in Z_n] for _ in range(world_size)]'''
        #gathered_aggregation = [[torch.zeros_like(param) for param in W_n] for _ in range(world_size)]
        
        
        
        current_step = 0
        avg_loss_list = []
        start_time = time.time()
        epoch_count = 0
        while True:        
            total_train_loss = 0
            
        
            model.train()  # `train` just changes the *mode* (train vs. eval), it doesn't *perform* the training.
        
            current_step += 4
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
        
                #batch_loss += loss.item()
                
                
        
                # Get sample every x batches.
                '''if step % 10 == 0 and not step == 0:
        
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))
        
                    #model.eval()
        
                    #sample_output = infer("eggs, flour, butter, sugar", model)
                    #print(sample_output)
        
                    # `train` just changes the *mode* (train vs. eval), it doesn't *perform* the training.
                    model.train()'''
        
                loss = loss / len(train_dataloader) 
                total_train_loss += loss.clone()
                loss.backward()
                
                    
            learning_rate_current =  get_linear_schedule_with_warmup_custom(current_step, warmup_steps, total_steps, learning_rate)
            #sigma_lr_current = (learning_rate/learning_rate_current)*sigma_lr
            sigma_lr_current = sigma_lr
            rho_lr_current = 1/learning_rate_current - sigma_lr_current
            gradients = [param.grad for param in model.parameters()]               
            with torch.no_grad():

                for i, (param_wn, param_pn, gradient, param_wg, accumulator) in enumerate(zip(W_n, P_n, gradients, model.parameters(), accumulators)):
                    accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) * (gradient+param_pn).pow(2)) 
                    #delta = 1 / (rho_lr + sigma_lr) * ((rho_lr_current+sigma_lr_current) * param_wg - (gradient + param_pn)/(torch.sqrt(accumulator) + epsilon))
                    delta = param_wg - 1 / (sigma_lr_current + rho_lr_current*(torch.sqrt(accumulator) + epsilon)) * (gradient + param_pn)
                    param_wn.copy_(delta.detach())
                    param_pn.add_(sigma_lr_current * (param_wn - param_wg))
                    
                #manual_adam_update(model, gradients, m, v, learning_rate_current, beta1, beta2, epsilon)
                    
                
            model.zero_grad()
            dist.all_reduce(total_train_loss, op=dist.ReduceOp.SUM)  # Sum the loss across all devices
            avg_loss = total_train_loss / world_size
            avg_loss_list.append(avg_loss.item())
            #print(f"Epoch {epoch_count}, Update count {update_count}. Average Loss: {avg_loss}")
            del loss, outputs, gradients
            
            with torch.no_grad():
                for i, param in enumerate(model.parameters()):
                    #W_n[i].copy_(alpha_n * ( W_n[i] + P_n[i] / (sigma_lr*(torch.sqrt(Z_n[i]) + epsilon))))
                    W_n[i].copy_(alpha_b[rank] * ( W_n[i] + P_n[i] / sigma_lr_current))
                    dist.all_reduce(W_n[i], op=dist.ReduceOp.SUM)
                    #dist.reduce(W_n[i], dst=0, op=dist.ReduceOp.SUM)
                    #dist.broadcast(W_n[i], src=0)
                    param.copy_(W_n[i].detach())
                    
            if rank == 0:  # Only the master rank (rank 0) will print the average loss
                print(f"Epoch {epoch_count}, Average Loss: {avg_loss.item():.4f}")
                
            epoch_count += 1
                        
        
        
        
        
            #if epoch_count==15:
                #dist.destroy_process_group()
            '''if epoch_count % 5 ==0 and epoch_count>0:
                print("")
                print("Running Validation...")
            
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
            
                print("  Validation Loss: {0:.2f}".format(avg_val_loss))'''
            if epoch_count > 425: 
                end_time = time.time()
                total_time = end_time - start_time
                #print(f"CUDA {rank} Total training time for {epoch_count} epochs: {total_time:.2f} seconds")
                
                total_time_tensor = torch.tensor([total_time], device=device)
        
                # Compute the minimum total_time across all GPUs
                dist.all_reduce(total_time_tensor, op=dist.ReduceOp.MIN)
                min_total_time = total_time_tensor.item()
                ### (51min, 01s)
                if min_total_time >3061:
                    print(f"Total training time for {epoch_count} epochs: {min_total_time:.2f} seconds")
                    if rank == 0:
                        np.save("avg_loss_DDAM_converged.npy", np.array(avg_loss_list))
                        #np.save("avg_loss_DDAM_previous.npy", np.array(avg_loss_list))
                    break
                #if rank == 0:
                    #print(f"Total training time for {epoch_count} epochs: {min_total_time:.2f} seconds")
                    #np.save("avg_loss_DDAM.npy", np.array(avg_loss_list))

    finally:
        print('kill process run for rank:', rank)
        dist.destroy_process_group()


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))
            

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


def main():

    model_name = "gpt2-medium"  # options: ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name,
                                                  clean_up_tokenization_spaces=True,
                                                  bos_token='<|startoftext|>',
                                                  eos_token='<|endoftext|>',
                                                  unk_token='<|unknown|>',
                                                  pad_token='<|pad|>'
                                                 )
    
    
    df_recipes = pd.read_csv('recipes_1000.csv')
    df_recipes.reset_index(drop=True, inplace=True)
    
    # df_recipes = df_recipes.iloc[:600]
    print(list(df_recipes.columns))
    print(f"data shape {df_recipes.shape}")
    data = df_recipes.apply(lambda x:form_string(
        x['ingredients'], x['instructions']), axis=1).to_list()
    
    doc_lengths = []
    
    for rec in df_recipes.itertuples():
    
        # get rough token count distribution
        tokens = nltk.word_tokenize(rec.ingredients + ' ' + rec.instructions)
    
        doc_lengths.append(len(tokens))
    
    doc_lengths = np.array(doc_lengths)
    # the max token length
    print(f"% documents > 180 tokens: {round(len(doc_lengths[doc_lengths > 180]) / len(doc_lengths) * 100, 1)}%")
    print(f"Average document length: {int(np.average(doc_lengths))}")
    
    
    # GPT2 is a large model. Increasing the batch size above 2 has lead to out of memory problems.
    batch_size = 16
    max_length = 180  # maximum sentence length
    
    dataset = RecipeDataset(data, tokenizer, max_length)
    
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
    accumulation_steps_per_update = []
    for ii in range(num_gpu-1):
        accumulation_steps_per_update.append(int(len(train_dataloader)/num_gpu))
        
    accumulation_steps_per_update.append(int(len(train_dataloader) - sum(accumulation_steps_per_update)))
    print('The accumulation_steps_per_update is: ', accumulation_steps_per_update)
    alpha_b = [i/sum(accumulation_steps_per_update) for i in accumulation_steps_per_update]
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
    
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    
    sigma_lr = 0.05 #0.08 #1000 suddenly increasiing # 9000 slow
    rho_lr = 10000 #10000 #1000 # 1000
    beta_rmsprop = 0.999
    rho = 0.9
    beta1 = 0.9
    beta2 = 0.999
    
    
    #sub_data_loaders = split_data_loader(train_dataloader, num_gpu, accumulation_steps_per_update)
    
    sub_data_loaders = []
    for ii in range(num_gpu):
        if ii != num_gpu-1:
            subset_dataloader_0 = itertools.islice(train_dataloader, ii*accumulation_steps_per_update[ii], (ii+1)*accumulation_steps_per_update[ii])
            sub_data_loaders.append(list(subset_dataloader_0))
        else:
            subset_dataloader_0 = itertools.islice(train_dataloader, ii*accumulation_steps_per_update[ii-1], ii*accumulation_steps_per_update[ii-1]+accumulation_steps_per_update[ii])
            sub_data_loaders.append(list(subset_dataloader_0))
            
    for ii in range(num_gpu):
        print(len(sub_data_loaders[ii]))
    
    world_size = num_gpu
    
    torch.multiprocessing.spawn(train, args=(world_size, sub_data_loaders, validation_dataloader, sigma_lr, rho_lr, beta_rmsprop, alpha_b), nprocs=world_size, join=True)
    

if __name__ == "__main__":
    main()