import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import time
import os
import collections


# Dataset defined by a set of transitions, actions, and rewards
class RL_Dataset(Dataset):
    
    def __init__(self, X, action, rewards, transition_dict, option = None):
        
#         if option == 'embed':
#             self.s0 = torch.LongTensor(s0) - 1
#             self.sp = torch.LongTensor(sp) - 1
#         elif option == 'linear':
#             self.s0 = torch.from_numpy(s0).float() - 1.0
#             self.sp = torch.from_numpy(sp).float() - 1.0
        
#             if len(self.s0.shape) == 1:
#                 self.s0.unsqueeze(1)
#                 self.sp.unsqueeze(1)
        
        self.X = torch.from_numpy(X).float()
        self.action = torch.LongTensor(action)
        self.rewards = torch.from_numpy(rewards).float()
        self.transition_dict = transition_dict
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx, ], self.action[idx], self.rewards[idx], self.X[self.transition_dict[idx], ]
    

# Dataset defined by a set of transitions, actions, and rewards
class transitionDataset(Dataset):
    def __init__(self, s0, action, rewards, sp, option = 'embed'):
        
        if option == 'embed':
            self.s0 = torch.LongTensor(s0) - 1
            self.sp = torch.LongTensor(sp) - 1
        elif option == 'linear':
            self.s0 = torch.from_numpy(s0).float() - 1.0
            self.sp = torch.from_numpy(sp).float() - 1.0
            
            if len(self.s0.shape) == 1:
                self.s0.unsqueeze(1)
                self.sp.unsqueeze(1)
        
        self.action = torch.LongTensor(action) - 1
        self.rewards = torch.from_numpy(rewards).float()

    def __len__(self):
        return self.s0.shape[0]
    
    def __getitem__(self, idx):
        return self.s0[idx], self.action[idx], self.rewards[idx], self.sp[idx]
    
# Dataset that is composed of indices of all possible states  
class evaluationDataset(Dataset):
    def __init__(self, num_states, option = 'embed', the_mean = None, the_std = None):
        
        if option == 'embed':
            self.states = torch.LongTensor(np.arange(num_states))     
        elif option == 'linear':
            if the_mean is None:
                self.states = torch.from_numpy(np.arange(num_states)).float()
            if (the_mean is not None) & (the_std is not None):
                self.states = torch.from_numpy((np.arange(num_states) - the_mean) / the_std).float()
        
    def __len__(self):
        return self.states.shape[0]
    
    def __getitem__(self, idx):
        return self.states[idx]
    
# A hidden linear layer with dropout, activation, and batch norm
class hidden_linear_layer(torch.nn.Module):
    
    def __init__(self, D_in, D_out, drop_prob):
        super(hidden_linear_layer, self).__init__()
        
        self.linear = nn.Linear(D_in, D_out).float()
        self.dropout = nn.Dropout(p = drop_prob)
        self.activation = nn.ELU()
#         print('D_out')
        self.batch_norm = nn.BatchNorm1d(num_features = D_out)
        
    def forward(self, x):
        result = self.dropout(self.activation(self.batch_norm(self.linear(x))))
        return result

# A feed forward net with arbitrary number of hidden layers - utilizes hidden_linear_layer
class feed_forward_net(torch.nn.Module):
    
    def __init__(self, D_in, H, D_out, embed_dim, drop_prob, num_hidden, option = 'linear'):
        # Initialize the network
        super(feed_forward_net, self).__init__()
        
        if option == 'embed':
            self.state_embed = nn.Embedding(D_in, embed_dim)
            self.input_layer = hidden_linear_layer(embed_dim, H, drop_prob)
            self.output_layer = nn.Linear(H, D_out)
            self.layers = nn.ModuleList([self.state_embed, self.input_layer])
        elif option == 'linear':
            self.input_layer = hidden_linear_layer(D_in, H, drop_prob)
            self.output_layer = nn.Linear(H, D_out)
            self.layers = nn.ModuleList([self.input_layer])
            
        if num_hidden > 1:
            self.layers.extend([hidden_linear_layer(H, H, drop_prob) for i in range(num_hidden - 1)])
            
        self.layers.append(self.output_layer)

    def forward(self, x):
        y_pred = nn.Sequential(*self.layers).forward(x)
        
        return y_pred

# A dueling feedforward net with arbitrary number of hidden layers
class dueling_net(torch.nn.Module):
    
    def __init__(self, D_in, H, D_out, drop_prob, num_hidden, option = 'linear'):
        # Initialize the network
        super(dueling_net, self).__init__()
        
        if option == 'embed':
            print('Doing Nothing!')
#             self.state_embed = nn.Embedding(D_in, embed_dim)
#             self.input_layer = hidden_linear_layer(embed_dim, H, drop_prob)
            
#             self.value_layer = nn.Linear(H, 1)
#             self.advantage_layer = nn.Linear(H, D_out)
#             self.layers = nn.ModuleList([self.state_embed, self.input_layer])
            
        elif option == 'linear':
            self.input_layer = hidden_linear_layer(D_in, H, drop_prob)
            self.value_layer = nn.Linear(H, 1)
            self.advantage_layer = nn.Linear(H, D_out)
            self.layers = nn.ModuleList([self.input_layer])
            
        if num_hidden > 1:
            self.layers.extend([hidden_linear_layer(H, H, drop_prob) for i in range(num_hidden - 1)])

    def forward(self, x):
        pre_output = nn.Sequential(*self.layers).forward(x)
        value = self.value_layer(pre_output)
        advantage = self.advantage_layer(pre_output)
        
        advantage_diff = advantage - advantage.mean(1, keepdim = True)
        
        y_pred = value + advantage_diff
        
        return y_pred

# Evaluates the model for all possible states
def evaluate_model(model, the_loader, dset_size, use_gpu = False, option = 'embed'):
    
    model.train(False)
    
    i = 0
    for states in the_loader:
        i = i + 1
        
        if option == 'linear':
            states = states.unsqueeze(1)
        
        # print(states.shape)
        
        if use_gpu:
            states  = Variable(states.cuda(async = True))
        else:
            states = Variable(states)
        
        outputs = model(states)
        predicted_actions = torch.max(outputs, 1)[1].data + 1 # Account for 0-1 indexing
        
        if i == 1:
            predicted_actions_list = predicted_actions
        else:
            predicted_actions_list = torch.cat((predicted_actions_list, predicted_actions))
    
    return predicted_actions_list

# # Trains model with default Q-learning
# def train_model(model, 
#                 loaders, 
#                 dset_sizes, 
#                 config, 
#                 criterion,
#                 optimizer,
#                 scheduler = None,
#                 use_gpu = False):
    
#     # Training loop
#     performance_dict = {
#                         'train': {
#                             'loss':[]
#                     } 
#                 }
    
#     since = time.time()
#     best_loss = 1e15

#     for epoch in range(config['num_epochs']):
#         print('-' * 10)
#         print('Epoch {}/{}'.format(epoch, config['num_epochs'] - 1))
#         print('-' * 10)

#         for phase in ['train']:

#             if phase == 'train':
#                 model.train(True)  # Set model to training mode
#             else:
#                 model.train(False)  # Set model to evaluate mode

#             running_loss = 0.0

#             i = 0

#             for the_data in loaders[phase]:
#                 i = i + 1
                
#                 states, actions, rewards, next_states = the_data
                
#                 batch_size = states.shape[0]
                
#                 if config['option'] == 'linear':
#                     states = states.unsqueeze(1)
#                     next_states = next_states.unsqueeze(1)
                
#                 if use_gpu:
#                     states, actions, rewards, next_states = Variable(states.cuda(async = True)), \
#                                                                Variable(actions.cuda(async = True)), \
#                                                                Variable(rewards.cuda(async = True)), \
#                                                                Variable(next_states.cuda(async = True),
#                                                                         volatile = True)
                            
#                 else:
#                     states, actions, rewards, next_states = Variable(states), \
#                                                             Variable(actions), \
#                                                             Variable(rewards), \
#                                                             Variable(next_states, volatile = True)
                
#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 outputs = model(states)
                
#                 if use_gpu:
#                     current_state_values = outputs[torch.LongTensor(np.arange(batch_size).tolist()).cuda(),
#                                                    actions.data]
#                     # current_state_values = outputs[np.arange(batch_size).tolist(),
#                     #                                actions.data]
#                 else:
#                     current_state_values = outputs[torch.LongTensor(np.arange(batch_size)),
#                                actions.data]
                
#                 next_state_values = torch.max(model(next_states), 1)[0]
            
#                 next_state_values.volatile = False
#                 expected_state_values = (next_state_values * config['gamma']) + rewards.float()

#                 loss = criterion(current_state_values, expected_state_values)
    
#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()

#                 running_loss += loss.data[0]

#             # Compute Metrics
#             epoch_loss = running_loss / dset_sizes[phase]

#             # Apply scheduler
#             if (scheduler is not None) & (phase == 'train'):
#                 scheduler.step(epoch_loss)

#             performance_dict[phase]['loss'].append(epoch_loss)
#             print('{} Loss: {:4f}'.format(phase, epoch_loss))

#             # deep copy the model
#             if (phase == 'train') & (epoch_loss < best_loss):
#                 print('Best Loss updated')
#                 best_loss = epoch_loss
#                 best_model_wts = model.state_dict()

#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val loss: {:4f}'.format(best_loss))
    
#     model.load_state_dict(best_model_wts)
#     return performance_dict, model, best_loss, time_elapsed

# # Trains model with separate target network
# def train_model_target(model,
#                         target_model,
#                         loaders, 
#                         dset_sizes, 
#                         config, 
#                         criterion,
#                         optimizer,
#                         # target_optimizer,
#                         scheduler = None,
#                         use_gpu = False):
    
#     # Training loop
#     performance_dict = {
#                         'train': {
#                             'loss':[]
#                     } 
#     }
#     since = time.time()
#     best_loss = 1e15

#     target_model.train(False)
    
#     for epoch in range(config['num_epochs']):
#         print('-' * 10)
#         print('Epoch {}/{}'.format(epoch, config['num_epochs'] - 1))
#         print('-' * 10)

#         for phase in ['train']:

#             if phase == 'train':
#                 model.train(True)  # Set model to training mode
#             else:
#                 model.train(False)  # Set model to evaluate mode

#             running_loss = 0.0

#             i = 0

#             for the_data in loaders[phase]:
#                 i = i + 1
                
#                 states, actions, rewards, next_states = the_data
                
#                 batch_size = states.shape[0]
                
                
#                 if config['option'] == 'linear':
#                     states = states.unsqueeze(1)
#                     next_states = next_states.unsqueeze(1)
                    
                
#                 if use_gpu:
#                     states, actions, rewards, next_states = Variable(states.cuda(async = True)), \
#                                                                Variable(actions.cuda(async = True)), \
#                                                                Variable(rewards.cuda(async = True)), \
#                                                                Variable(next_states.cuda(async = True), volatile = True)
                            
#                 else:
#                     states, actions, rewards, next_states = Variable(states), \
#                                                             Variable(actions), \
#                                                             Variable(rewards), \
#                                                             Variable(next_states, volatile = True)

#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#                 # target_optimizer.zero_grad()

#                 # forward
#                 outputs = model(states) # returns batch_size x action_dim
                
#                 if use_gpu:
#                     current_state_values = outputs[torch.LongTensor(np.arange(batch_size).tolist()).cuda(),
#                                                    actions.data]
#                 else:
#                     current_state_values = outputs[torch.LongTensor(np.arange(batch_size)),
#                                actions.data]
                
#                 next_state_values = torch.max(target_model(next_states), 1)[0]
#                 next_state_values.volatile = False
#                 expected_state_values = (next_state_values * config['gamma']) + rewards.float()

#                 loss = criterion(current_state_values, expected_state_values)
    
#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()
#                     # target_optimizer.step()

#                 running_loss += loss.data[0]
            
#             if (epoch % config['target_update'] == 0) & (epoch > 1):
#                 print('Updating Target Model')
#                 # target_model.load_state_dict(best_model_wts)
#                 target_model.load_state_dict(model.state_dict())
            
#             # Compute Metrics
#             epoch_loss = running_loss / dset_sizes[phase]

#             # Apply scheduler
#             if (scheduler is not None) & (phase == 'train'):
#                 scheduler.step(epoch_loss)

#             performance_dict[phase]['loss'].append(epoch_loss)
#             print('{} Loss: {:4f}'.format(phase, epoch_loss))

#             # deep copy the model
#             if (phase == 'train') & (epoch_loss < best_loss):
#                 print('Best Loss updated')
#                 best_loss = epoch_loss
#                 best_model_wts = model.state_dict()

#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val loss: {:4f}'.format(best_loss))
    
#     # model.load_state_dict(best_model_wts)
#     return performance_dict, model, best_loss, time_elapsed

# Trains model with Double Q-Learning
def train_model_double(model,
                        target_model,
                        loaders, 
                        dset_sizes, 
                        config, 
                        criterion,
                        optimizer,
                        # target_optimizer,
                        scheduler = None,
                        use_gpu = False):
    
    # Training loop
    performance_dict = {
                        'train': {
                         'loss':[]
                    } 
    }
    since = time.time()
    best_loss = 1e15
    best_value = -1.0*best_loss
    
    target_model.train(False)
    
    
    for epoch in range(config['num_epochs']):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, config['num_epochs'] - 1))
        print('-' * 10)

        for phase in ['train']:

            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_value = 0.0

            i = 0

            for the_data in loaders[phase]:
                i = i + 1
                
                states, actions, rewards, next_states = the_data
#                 print(states)
#                 print(actions)
#                 print(rewards)
#                 print(next_states)
                
                batch_size = states.shape[0]
                
#                 if config['option'] == 'linear':
#                     states = states.unsqueeze(1)
#                     next_states = next_states.unsqueeze(1)
                
                
                if use_gpu:
                    states, actions, rewards, next_states = Variable(states.cuda(async = True)), \
                                                               Variable(actions.cuda(async = True)), \
                                                               Variable(rewards.cuda(async = True)), \
                                                               Variable(next_states.cuda(async = True), volatile = True)
                            
                else:
                    states, actions, rewards, next_states = Variable(states), \
                                                            Variable(actions), \
                                                            Variable(rewards), \
                                                            Variable(next_states, volatile = True)
                
                # Compute action primes
#                 print(states)
#                 print(actions)
#                 print(rewards)
#                 print(next_states)
                temp = model(next_states)
#                 print(temp)
                
                action_prime = torch.max(model(next_states), 1)[1].data
                
                # Get target_outputs
                target_ouput = target_model(next_states)

                # Get the value of the next states at the action_prime
                if use_gpu:
                    next_state_values = target_ouput[torch.LongTensor(np.arange(batch_size).tolist()).cuda(),
                                                   action_prime]
                else:
                    next_state_values = target_ouput[torch.LongTensor(np.arange(batch_size)),
                               action_prime]
                    
                next_state_values.volatile = False
                expected_state_values = (next_state_values * config['gamma']) + rewards.float()
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(states)
                
                if use_gpu:
                    current_state_values = outputs[torch.LongTensor(np.arange(batch_size).tolist()).cuda(),
                                                   actions.data]
                    
                else:
                    current_state_values = outputs[torch.LongTensor(np.arange(batch_size)),
                               actions.data]
                    
                loss = criterion(current_state_values, expected_state_values)
    
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_value += current_state_values.data.sum()
                
            if epoch % config['target_update'] == 0:
                print('Updating Target Model')
                target_model.load_state_dict(model.state_dict())
            
            # Compute Metrics
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_value = running_value / dset_sizes[phase]

            # Apply scheduler
            if (scheduler is not None) & (phase == 'train'):
                scheduler.step(epoch_loss)

            performance_dict[phase]['loss'].append(epoch_loss)
            print('{} Loss: {:4f}, Value: {:4f}'.format(phase, epoch_loss, epoch_value))

            # deep copy the model
            if (phase == 'train') & (epoch_value > best_value):
                print('Best Value updated')
                best_value = epoch_value
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best value: {:4f}'.format(best_value))
    
    model.load_state_dict(best_model_wts)
    return performance_dict, model, best_loss, time_elapsed

# Overwrites part of dataLoader code - slight modification from the PyTorch Github code
# to account for issues with Double vs Float
def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif type(batch[0]).__module__ == 'numpy':
        elem = batch[0]
        if type(elem).__name__ == 'ndarray':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)
#     elif isinstance(batch[0], string_classes):
#         return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [my_collate(samples) for samples in transposed]

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))