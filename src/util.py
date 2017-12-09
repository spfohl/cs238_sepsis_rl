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
        
        self.X = torch.from_numpy(X).float()
        self.action = torch.LongTensor(action)
        self.rewards = torch.from_numpy(rewards).float()
        self.transition_dict = transition_dict
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx, ], self.action[idx], self.rewards[idx], self.X[self.transition_dict[idx], ]
    
# Dataset defined by a set states and actions for estimating the Q
class evaluation_Dataset(Dataset):
    
    def __init__(self, X, action, option = None):
        
        self.X = torch.from_numpy(X).float()
        self.action = torch.LongTensor(action)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx, ], self.action[idx]
    
# A hidden linear layer with dropout, activation, and batch norm
class hidden_linear_layer(torch.nn.Module):
    
    def __init__(self, D_in, D_out, drop_prob):
        super(hidden_linear_layer, self).__init__()
        
        self.linear = nn.Linear(D_in, D_out).float()
        self.dropout = nn.Dropout(p = drop_prob)
        self.activation = nn.ELU()
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
        
        if option == 'linear':
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
                                 'loss':[],
                                 'value': [],
                                 'empirical_value': []
                                 },
                        'val': {
                                 'loss':[],
                                 'value': [],
                                 'empirical_value': []
                                 } 
                        }
    
    since = time.time()
    best_loss = 1e15
    best_value = -1.0*best_loss
    best_empirical_value = best_value
    
    target_model.train(False)
    
    
    for epoch in range(config['num_epochs']):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, config['num_epochs'] - 1))
        print('-' * 10)

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_value = 0.0
            running_empirical_value = 0.0

            i = 0

            for the_data in loaders[phase]:
                i = i + 1
                
                states, actions, rewards, next_states = the_data
                
                batch_size = states.shape[0]
                
                
                
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
                
                # Compute action primes - plug in next states to main model
                action_prime = torch.max(model(next_states), 1)[1].data

                # Get target_outputs - plug in next states to target model
                target_ouput = target_model(next_states)

                # Get the value of the next states at the action_prime from the target model
                if use_gpu:
                    next_state_values = target_ouput[torch.LongTensor(np.arange(batch_size).tolist()).cuda(),
                                                   action_prime]
                else:
                    next_state_values = target_ouput[torch.LongTensor(np.arange(batch_size)),
                               action_prime]

                next_state_values.volatile = False
                reward_mask = rewards == 0
                expected_state_values = (next_state_values * config['gamma']) * reward_mask.float() + rewards.float()            
                
                if phase == 'train':
                    # zero the parameter gradients
                    optimizer.zero_grad()

                # Compute the Q-values for the model at the current state values at the actions actually taken
                outputs = model(states)
                
                # Get the best possible actions from the current model
                best_actions = torch.max(outputs, 1)[1].data
                
                if use_gpu:
                    # Get Q(s, a)
                    current_state_values = outputs[torch.LongTensor(np.arange(batch_size).tolist()).cuda(),
                                                   actions.data]
                    
                    # Get Q(s, a*)
                    best_policy_values = outputs[torch.LongTensor(np.arange(batch_size).tolist()).cuda(),
                                                   best_actions]
                    
                else:
                    # Get Q(s, a)
                    current_state_values = outputs[torch.LongTensor(np.arange(batch_size)),
                                                   actions.data]
                    
                    # Get Q(s, a*)
                    best_policy_values = outputs[torch.LongTensor(np.arange(batch_size)),
                                                   best_actions]
                    
                loss = criterion(current_state_values, expected_state_values)
    
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.data[0]
                running_value += best_policy_values.data.sum()
                running_empirical_value += current_state_values.data.sum()
                
            if (epoch % config['target_update'] == 0) & (phase == 'train'):
                print('Updating Target Model')
                target_model.load_state_dict(model.state_dict())
            
            # Compute Metrics
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_value = running_value / dset_sizes[phase]
            epoch_empirical_value = running_empirical_value / dset_sizes[phase]

            # Apply scheduler
            if (scheduler is not None) & (phase == 'train'):
                scheduler.step(epoch_loss)

            # Update the performance_dict
            performance_dict[phase]['loss'].append(epoch_loss)
            performance_dict[phase]['value'].append(epoch_value)
            performance_dict[phase]['empirical_value'].append(epoch_empirical_value)
            
            print('{} Loss: {:4f}, Best Value: {:4f}, Empirical Value: {:4f}'.format(phase, 
                                                                                     epoch_loss, 
                                                                                     epoch_value, 
                                                                                     epoch_empirical_value))

            # Copy the state_dict
            if (phase == 'val') & (epoch_value > best_value):
                print('Best Value updated')
                best_value = epoch_value
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best value: {:4f}'.format(best_value))
    
    # Load the best model seen so far
    model.load_state_dict(best_model_wts)
    return performance_dict, model, best_loss, time_elapsed

# Converts a performance_dict returned from a model 
def dict_to_df(the_dict):
    
    df_list = []
    for phase, phase_dict in the_dict.items():
        
        metrics = phase_dict.keys()
        
        phase_df = pd.DataFrame.from_dict(phase_dict)
        
        phase_df['phase'] = phase
        phase_df['epoch'] = np.arange(phase_df.shape[0])
        
        phase_df = phase_df.melt(
                                id_vars = ['phase', 'epoch'], 
                                 value_vars = phase_dict.keys(), 
                                 var_name = 'Metric', 
                                 value_name = 'Performance')
        df_list.append(phase_df)
        
    return(pd.concat(df_list))

# For evaluating a model
def evaluate_model(model, the_loader, use_gpu = False):
    
    model.train(False)
    i = 0
    for states, actions in the_loader:
        i = i + 1
        batch_size = states.shape[0]
        
        if use_gpu:
            states, actions = Variable(states.cuda(async = True)), \
                                Variable(actions.cuda(async = True))
                
        else:
            states, actions = Variable(states), \
                                Variable(actions)
                
        outputs = model(states)
        best_actions = torch.max(outputs, 1)[1].data
        
        if use_gpu:
            # Get Q(s, a)
            current_state_values = outputs[torch.LongTensor(np.arange(batch_size).tolist()).cuda(),
                                           actions.data]
            # Get Q(s, a*)
            best_policy_values = outputs[torch.LongTensor(np.arange(batch_size).tolist()).cuda(),
                                        best_actions]
                
        else:
            # Get Q(s, a)
            current_state_values = outputs[torch.LongTensor(np.arange(batch_size)),
                                           actions.data]
            # Get Q(s, a*)
            best_policy_values = outputs[torch.LongTensor(np.arange(batch_size)),
                                                   best_actions]
            
        if i == 1:
            predicted_state_action_values_list = current_state_values
            predicted_state_action_values_list_best = best_policy_values
            best_actions_list = best_actions
        else:
            predicted_state_action_values_list = torch.cat((predicted_state_action_values_list, 
                                                            current_state_values))
            predicted_state_action_values_list_best = torch.cat((predicted_state_action_values_list_best, 
                                                            best_policy_values))
            best_actions_list = torch.cat((best_actions_list, 
                                            best_actions))
    
    return predicted_state_action_values_list, predicted_state_action_values_list_best, best_actions_list