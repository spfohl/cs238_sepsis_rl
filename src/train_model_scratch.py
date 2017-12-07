
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk
import os
import torch
import copy
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from sklearn.model_selection import ParameterGrid


from functools import reduce
from util import *


# In[2]:


# %load_ext autoreload
# %autoreload 2


# In[3]:


use_gpu = torch.cuda.is_available()
print(use_gpu)


# In[4]:


data_path = '../data'
data_dict = joblib.load(os.path.join(data_path, 'data_dict.pkl'))

exp_name = 'exp2'


# In[5]:


# Key = starting state, value = next state
transition_dict_train = dict(zip(data_dict['train']['state_id'], data_dict['train']['next_state_id']))
transition_dict_val = dict(zip(data_dict['val']['state_id'], data_dict['val']['next_state_id']))


# In[6]:


# Define the tuning grid
param_options = {
                  'state_dim' : [data_dict['train']['X'].shape[1]],
                  'action_dim' : [25],
                  'gamma' : [0.9],
                  'batch_size' : [512],
                  'lr' : [1e-4],
                  'num_epochs' : [300],
                  'hidden_dim' : [128, 256, 512, 1024],
                  'num_hidden' : [1, 2, 3, 5, 10],
                  'drop_prob' : [0.0],
                  'target_update': [10],
                  'option' : ['linear'],
                  'use_scheduler' : [False]
         }

config_grid = ParameterGrid(param_options)


# In[7]:


# Train all the models
for config in config_grid:
    
    # Create a Dataset
    train_dataset = RL_Dataset(data_dict['train']['X'], 
                               data_dict['train']['action'],
                               data_dict['train']['reward'],
                               transition_dict_train)

    val_dataset = RL_Dataset(data_dict['val']['X'], 
                               data_dict['val']['action'],
                               data_dict['val']['reward'],
                               transition_dict_val)
    # Create a dataloader
    train_loader = DataLoader(train_dataset, 
                            config['batch_size'],
                            shuffle = True,
                            num_workers = 32
                             )

    val_loader = DataLoader(val_dataset, 
                             config['batch_size'],
                             shuffle = True,
                             num_workers = 32
                             )

    loaders = {'train' : train_loader,
               'val' : val_loader
              }

    dset_sizes = {'train' : len(train_dataset),
                  'val' : len(val_dataset)
                 }
    
    
    
    print(config)
    model = dueling_net(D_in = config['state_dim'], 
                        H = config['hidden_dim'], 
                        D_out = config['action_dim'],
                        drop_prob = config['drop_prob'],
                        num_hidden = config['num_hidden'],
                        option = config['option']
                       )

    target_model = dueling_net(D_in = config['state_dim'], 
                                H = config['hidden_dim'], 
                                D_out = config['action_dim'],
                                drop_prob = config['drop_prob'],
                                num_hidden = config['num_hidden'],
                                option = config['option']
                              )

    optimizer = optim.Adam([{'params': model.parameters()}], 
                            lr = config['lr'])

    if config['use_scheduler']:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', verbose = True)
    else:
        scheduler = None

    def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight.data).float()

    model.apply(weights_init)
    target_model.apply(weights_init)

    if use_gpu:
        model = model.cuda()
        target_model.cuda()

    criterion = torch.nn.SmoothL1Loss(size_average = False)

    performance_dict, best_model, best_loss, time_elapsed = train_model_double(model = model, 
                                                                                target_model = target_model,
                                                                                loaders = loaders, 
                                                                                dset_sizes = dset_sizes, 
                                                                                config = config, 
                                                                                criterion = criterion,
                                                                                optimizer = optimizer,
                                                                                scheduler = scheduler,
                                                                                use_gpu = use_gpu)
    
    config_str = reduce(lambda x, y: x + y + '_', [str(key) + '_' + str(value) for key, value in config.items()]) + 'time_'+ str(time.time()).split('.')[1]
    checkpoints_path = os.path.join('../checkpoints/' + exp_name, config_str + '.chk')
    torch.save(best_model.state_dict(), checkpoints_path)
    
    performance_df = dict_to_df(performance_dict)
    config_df = pd.DataFrame(config, index = [0])
    config_df['config_str'] = config_str
    performance_df['config_str'] = config_str
    performance_df = performance_df.set_index('config_str').join(config_df.set_index('config_str'))
    performance_df.to_csv(os.path.join('../performance/' + exp_name, 'performance_' + config_str + '.csv'), index = True)

