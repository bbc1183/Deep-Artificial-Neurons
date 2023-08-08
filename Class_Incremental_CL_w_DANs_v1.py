from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
from torchmeta.datasets.helpers import omniglot
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchmeta.utils.data import BatchMetaDataLoader
from torch.autograd import Variable
from collections import Counter
import time
import pandas
import csv
import copy
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image, ImageOps, ImageFilter
import random
from torchvision.utils import save_image
import higher


"""
EXPERIMENT_16

- non-iid class incremental hyper-learning base file
- gpu-memory optimized 
-   


"""

#-----------------------------------------------------------------------------------------------------------------------------
#  DATA AUGMENTATION STUFF 
def exclude_bias_and_norm(p):
    return p.ndim == 1


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Single_Channel_Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(32, interpolation=Image.BICUBIC),
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(45, -45)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))

        ])
        self.transform_prime = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(32, interpolation=Image.BICUBIC),
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(45, -45)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))

        ])
        self.actual_transform=transforms.Compose(
                    [  
                        transforms.ToPILImage(),
                        transforms.RandomResizedCrop(32, interpolation=Image.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __call__(self, x):
        
        actual = self.actual_transform(x)
        
        
        augmented = self.transform_prime(x)
        # augmented_prime = self.transform_prime(x)
        
        
        return augmented
 
#-----------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------
class Simple_DAN(nn.Module):
    #original dims: [n_channels, 15, 8, 1]
    def __init__(self, n_channels):
        super(Simple_DAN, self).__init__()
        self.conv1 = nn.Conv2d( 1,16,2,1)  
        self.conv2 = nn.Conv2d(16,8,2,1)  

        self.pool = nn.MaxPool2d(2, stride=1)
        self.fc2 = nn.Linear(512, 4000)
        self.fc3 = nn.Linear(4000, 30)
        self.fc4 = nn.Linear(30, 1)  
        self.dropout = nn.Dropout(0.3)
      
        self.batch_size = 5
    def forward(self, x, batch_size=5, DAN_dropout = False): 
        
        x = self.conv1(x)
        x = F.leaky_relu(x)
        if DAN_dropout == True: 
            x = self.dropout(x)
       
        x = self.conv2(x)
        x = F.leaky_relu(x)
        if DAN_dropout == True: 
            x = self.dropout(x)
       
        x = torch.flatten(x, 1)
        x2 = F.leaky_relu(self.fc2(x))
        if DAN_dropout == True: 
            x = self.dropout(x)
       
        x3 = F.leaky_relu(self.fc3(x2))
        x4 =F.leaky_relu( self.fc4(x3))
        x4 = x4.view(batch_size, int(x4.shape[0]/batch_size))
        return x4

class DAN_classifier(nn.Module):

    def __init__(self, dims, n_channels):
        super(DAN_classifier, self).__init__()   
        self.dims = dims 
        self.num_DAN_layers = len(dims)-1
        self.n_channels = n_channels 
        self.VECS_layers = nn.ModuleDict() 
        self.DAN_layers = nn.ModuleDict() 
        self.pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.3)
        #self.DAN = Simple_DAN(n_channels)
        self.dan_acts = torch.Tensor().cpu()
        self.DAN_layers["conv1"] = nn.Conv2d( 3,32,3,1, padding=1)
        self.DAN_layers["conv2"] = nn.Conv2d( 32, 4, 3, 1, padding=1)
        #self.DAN_layers["BN_conv2"] = nn.BatchNorm2d(4)

        
        self.VECS_layers["encoder1"] = nn.Linear(256,self.dims[1]*n_channels) #nn.Linear(self.dims[0],self.dims[1]*n_channels)
        for e_l in range(2,len(dims)-1):
            self.VECS_layers["encoder"+str(e_l)] = nn.Linear(self.dims[e_l-1],self.dims[e_l]*n_channels)
        
        master_synapse_input_dim = 256
        for d in range(1, len(dims)-1):
            master_synapse_input_dim += self.dims[d]
        self.VECS_layers["master_synapses"] = nn.Linear(master_synapse_input_dim, self.dims[-1]*n_channels) 
       
        for d_l in range(1, len(dims)):
            self.DAN_layers["DAN_phenotype_"+str(d_l)] = Simple_DAN(n_channels)


    def forward(self, x, batch_size=5, use_dropout=False, DAN_dropout=False, train_step = False):
       
        if train_step == True:
            batch_size = x.shape[0]
            skips = []
            x = self.DAN_layers["conv1"](x)
            x = F.leaky_relu(x)
            x = self.pool(x)
            x = self.DAN_layers["conv2"](x)
            x = F.leaky_relu(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            skips.append(x.clone())


            for l in range(1,len(self.dims)-1):
                x = self.VECS_layers["encoder"+str(l)].to("cuda:0")(x.to("cuda:0")).to("cuda:0")
                x = x.view((x.size()[0]*self.dims[l], 1, int(self.n_channels**.5),int(self.n_channels**.5)))
                x = F.leaky_relu(x)
                x = self.DAN_layers["DAN_phenotype_"+str(l)](x, batch_size, DAN_dropout)
                if l < len(self.dims)-2:
                    skips.append(x.clone())

            #take a look at all the activations throughout the network
            for skip in skips:
                x = torch.cat((x, skip), dim = 1)
              
            x = self.VECS_layers["master_synapses"].to("cuda:0")(x.to("cuda:0")).to("cuda:0")
            x = x.view((x.size()[0]*self.dims[-1], 1, int(self.n_channels**.5),int(self.n_channels**.5)))
            x = F.leaky_relu(x)
            x = self.DAN_layers["DAN_phenotype_"+str(len(self.dims)-1)](x,batch_size, DAN_dropout)
            
            x = F.log_softmax(x, dim=1)

        else:
            batch_size = x.shape[0]
            skips = []
            x = self.DAN_layers["conv1"](x)
            x = F.leaky_relu(x)
            x = self.pool(x)
            x = self.DAN_layers["conv2"](x)
            x = F.leaky_relu(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            skips.append(x.clone())


            for l in range(1,len(self.dims)-1):
                x = self.VECS_layers["encoder"+str(l)](x)
                x = x.view((x.size()[0]*self.dims[l], 1, int(self.n_channels**.5),int(self.n_channels**.5)))
                x = F.leaky_relu(x)
                x = self.DAN_layers["DAN_phenotype_"+str(l)](x, batch_size, DAN_dropout)
                if l < len(self.dims)-2:
                    skips.append(x.clone())

            #take a look at all the activations throughout the network
            for skip in skips:
                x = torch.cat((x, skip), dim = 1)
              
            x = self.VECS_layers["master_synapses"](x)
            x = x.view((x.size()[0]*self.dims[-1], 1, int(self.n_channels**.5),int(self.n_channels**.5)))
            x = F.leaky_relu(x)
            x = self.DAN_layers["DAN_phenotype_"+str(len(self.dims)-1)](x,batch_size, DAN_dropout)

            x = F.log_softmax(x, dim=1)
        


        return x
#---------------------------------------------------------------------------------------
num_models = 3
k_streams = 3
samples_per_stream = 19
samples_per_stream_multiplier = 32
my_batch_size = 5
hyper_batch_end = samples_per_stream*samples_per_stream_multiplier
hyper_batch_size = 5
samples_per_task = my_batch_size
num_tasks = 5
test_tasks = num_tasks
num_hyper_train_samples_per_task = 5
num_hyper_train_CL_tasks = 10
CL_task_sequence_length = 10
num_hyper_train_tasks = num_hyper_train_CL_tasks+110
num_img_tasks = 10
k_steps = 25
test_steps = 25
num_hyper_steps = 1
dan_lr = .0005
vecs_lr = .0005
num_cycles = 10

eval_samples_per_stream = 50
eval_samples_per_task = 5


transform=transforms.Compose([
        
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])


def get_Fashion_MNIST_Hyper_Stream(ways, shots):
    #----- fashion mnist stuff ----------------------------------
    torch.manual_seed(np.random.randint(1000))
    trainset = datasets.FashionMNIST('../data', download = True, train = True, transform = transform)
    testset = datasets.FashionMNIST('../data', download = True, train = False, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 700, shuffle = True)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 700, shuffle = True)

    stream_images = {}
    stream_targets = {}

    
    if ways <= 10:
        tasks = torch.Tensor(np.random.choice(range(10), ways, replace=False))
    else:
        extra = ways-10
        tasks = torch.Tensor(np.random.choice(range(10), 10, replace=False))
        extra = torch.Tensor(np.random.choice(range(10), extra, replace=False))
        tasks = torch.cat((tasks, extra))
    adjusted_tasks = torch.Tensor(np.random.choice(range(num_hyper_train_tasks), ways, replace=False))
    for t in tasks:
        t_idx = int(t.item())
        stream_images[str(t_idx)] = torch.Tensor()
        stream_targets[str(t_idx)] = torch.Tensor()
    
    while True:   
        dataiter = iter(trainloader)
        for batch_idx in range(len(dataiter)):
            images, labels = dataiter.next() #get a batch of images (tasks)
            for img_idx, img in enumerate(images): #for each image in the batch 
                t_idx = int(labels[img_idx].item())
                if t_idx in tasks: #check if the label is one of the selected tasks 
                    adjusted_task_idx = (tasks==t_idx).nonzero().item()
                    # if it is, check if we still need to 'fill up' this task 
                    if len(stream_images[str(t_idx)]) < shots:
                        stream_images[str(t_idx)] = torch.cat((stream_images[str(t_idx)], img.unsqueeze(0)), dim=0)
                        
                        stream_targets[str(t_idx)] = torch.cat((stream_targets[str(t_idx)], torch.Tensor([adjusted_task_idx])))
                
                    return_flag = True
                    for t in tasks:
                        idx = int(t.item())
                        if len(stream_images[str(idx)]) < shots:
                            return_flag = False 
                    if return_flag == True:
                        img_stream = torch.Tensor()
                        target_stream = torch.Tensor()
                        for b in stream_images.keys():
                            img_stream = torch.cat((img_stream, stream_images[b]))
                            target_stream = torch.cat((target_stream, stream_targets[b]))
                        return (img_stream, target_stream)

def get_CIFAR100_Hyper_Stream(ways, shots):
    #----- fashion mnist stuff ----------------------------------
    torch.manual_seed(np.random.randint(1000))
    trainset = datasets.CIFAR100('../data', download = True, train = True, transform = transform)
    testset = datasets.CIFAR100('../data', download = True, train = False, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 700, shuffle = True)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 700, shuffle = True)

    stream_images = {}
    stream_targets = {}

    

    tasks = torch.Tensor(np.random.choice(range(100), ways, replace=False))
    adjusted_tasks = torch.Tensor(np.random.choice(range(num_hyper_train_tasks), ways, replace=False))
    for t in tasks:
        t_idx = int(t.item())
        stream_images[str(t_idx)] = torch.Tensor()
        stream_targets[str(t_idx)] = torch.Tensor()
    while True:
        dataiter = iter(trainloader)
        for batch_idx in range(len(dataiter)):
            images, labels = dataiter.next() #get a batch of images (tasks)
            for img_idx, img in enumerate(images): #for each image in the batch 
                t_idx = int(labels[img_idx].item())
                if t_idx in tasks: #check if the label is one of the selected tasks 
                    adjusted_task_idx = (tasks==t_idx).nonzero().item()
                    # if it is, check if we still need to 'fill up' this task 
                    if len(stream_images[str(t_idx)]) < shots:
                        stream_images[str(t_idx)] = torch.cat((stream_images[str(t_idx)], img.unsqueeze(0)), dim=0)
                        
                        stream_targets[str(t_idx)] = torch.cat((stream_targets[str(t_idx)], torch.Tensor([adjusted_task_idx])))
                
                    return_flag = True
                    for t in tasks:
                        idx = int(t.item())
                        if len(stream_images[str(idx)]) < shots:
                            return_flag = False 
                    if return_flag == True:
                        img_stream = torch.Tensor()
                        target_stream = torch.Tensor()
                        for b in stream_images.keys():
                            img_stream = torch.cat((img_stream, stream_images[b]))
                            target_stream = torch.cat((target_stream, stream_targets[b]))
                        return (img_stream, target_stream)







def reset_models(models, dims, n_channels, device, batch_idx):
    for model_idx, model_key in enumerate(models.keys()):
        #--- MODEL RELOAD  (not necessary if learning a prior over the whole model) ------------------------------------
        
        gc.collect()
        torch.cuda.empty_cache()

        torch.manual_seed(torch.randint(0,400, (1,)).item())

        
        #models[model_key]["model"] = DAN_classifier(dims=dims, n_channels=n_channels)
        
        if batch_idx == 0:
            # phenotype_model = DAN_classifier(dims=[32*32, 400, 200, 200, 200, 200, num_hyper_train_tasks], n_channels=n_channels)
            # phenotype_model.load_state_dict(torch.load("noniid_hyper_learning16_large_m1.pt"))
            # for n, p in phenotype_model.named_parameters():
            #     for n2, p2 in models[model_key]["model"].named_parameters():
            #         if ((n2 == n) or ("DAN_phenotype_6" in n and "DAN_phenotype_"+str(dims[-1]) in n2)) and p.shape == p2.shape: 
            #             p2.data = p.data 
            phenotype_model = DAN_classifier(dims=dims, n_channels=n_channels)
            phenotype_model.load_state_dict(torch.load("noniid_hyper_learning16_ultra_m1.pt"))
            #models[model_key]["model"].DAN_layers.load_state_dict(phenotype_model.DAN_layers.state_dict())
            #models[model_key]["dan_opt"] = torch.optim.Adam(models[model_key]["model"].DAN_layers.parameters(),  lr=dan_lr)
            models[model_key]["model"].load_state_dict(phenotype_model.state_dict())       
                    
        else:
            phenotype_model = DAN_classifier(dims=dims, n_channels=n_channels)
            #phenotype_model.load_state_dict(torch.load("noniid_hyper_learning16_ultra2_m1.pt"))
            #models[model_key]["model"].DAN_layers.load_state_dict(phenotype_model.DAN_layers.state_dict())
            #models[model_key]["dan_opt"] = torch.optim.Adam(models[model_key]["model"].DAN_layers.parameters(),  lr=dan_lr)
            models[model_key]["model"].VECS_layers.load_state_dict(phenotype_model.VECS_layers.state_dict())

        
        models[model_key]["vecs_opt"] = torch.optim.Adam(models[model_key]["model"].VECS_layers.parameters(),  lr=vecs_lr)
        
        #models[model_key]["dan_opt"] = torch.optim.Adam(models[model_key]["model"].DAN_layers.parameters(),  lr=dan_lr)
        
        
        
        #--- END - ORIGINAL MODEL RELOAD ------------------------------------
    return models 

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 

#-----------------------------------------------------------------------------------------------------------------------------------
# for evaluating performance on the meta-training data 
def get_meta_train_memory_loss(models, data, target, cur_task_idx, device):
    global samples_per_task
    global samples_per_stream

    
    for m_idx in range(num_models):
        models["m"+str(m_idx+1)]["dan_opt"].zero_grad()
    

    
    
    
    mem_accs = []
    for m_idx in range(num_models):
        models["m"+str(m_idx+1)]["model"].cpu()
        memory_loss = 0
        for mem_sample in range(cur_task_idx+1):
            batch_start = mem_sample*samples_per_stream
            data[m_idx,batch_start:batch_start+samples_per_task].reshape((samples_per_task,3,32,32))
            cur_mem_data = data[m_idx,batch_start:batch_start+samples_per_task].reshape((samples_per_task,3,32,32)) #data[:, batch_start:batch_start+samples_per_task].reshape((my_batch_size*samples_per_task,3,32,32))
            cur_mem_targets = target[m_idx, batch_start:batch_start+samples_per_task].reshape((samples_per_task))
            cur_mem_targets = cur_mem_targets.long()
            cur_mem_data = cur_mem_data.cpu()
            cur_mem_targets = cur_mem_targets.cpu()
            cur_mem_output = models["m"+str(m_idx+1)]["model"](cur_mem_data, samples_per_task, use_dropout=False)
            cur_mem_output = torch.reshape(cur_mem_output[:,:num_hyper_train_tasks], (cur_mem_output.size()[0],num_hyper_train_tasks))
           
            memory_loss += F.nll_loss(cur_mem_output, cur_mem_targets) #F.mse_loss(cur_mem_output, cur_mem_targets)
        memory_loss /= cur_task_idx+1
        mem_accs.append(memory_loss.item())
        models["m"+str(m_idx+1)]["model"].cpu()
   
    memory_loss = np.mean(mem_accs)
    memory_std = np.std(mem_accs)
    
    return memory_loss, memory_std


#-----------------------------------------------------------------------------------------------------------------------------------
#for evaluation on a held out dataset (MNIST FOR NOW)
def get_meta_eval_memory_loss(models, data, target, cur_task_idx):
    global eval_samples_per_task
    global eval_samples_per_stream

    samples_per_task = eval_samples_per_task
    samples_per_stream = eval_samples_per_stream

    for m_idx in range(num_models):
        models["m"+str(m_idx+1)]["dan_opt"].zero_grad()

    mem_accs = []
    for m_idx in range(num_models):
        memory_loss = 0
        for mem_sample in range(cur_task_idx+1):
            batch_start = mem_sample*eval_samples_per_stream
            data[m_idx,batch_start:batch_start+eval_samples_per_task].reshape((eval_samples_per_task,3,32,32))
            cur_mem_data = data[m_idx,batch_start:batch_start+eval_samples_per_task].reshape((eval_samples_per_task,3,32,32)) #data[:, batch_start:batch_start+samples_per_task].reshape((my_batch_size*samples_per_task,3,32,32))
            cur_mem_targets = target[m_idx, batch_start:batch_start+eval_samples_per_task].reshape((eval_samples_per_task))
            cur_mem_targets = cur_mem_targets.long()
            cur_mem_output = models["m"+str(m_idx+1)]["model"](cur_mem_data, eval_samples_per_task, use_dropout=False)
            cur_mem_output = torch.reshape(cur_mem_outputcur_mem_output[:,:num_tasks], (cur_mem_output.size()[0],num_tasks))
            memory_loss += F.nll_loss(cur_mem_output, cur_mem_targets) #F.mse_loss(cur_mem_output, cur_mem_targets)
        memory_loss /= cur_task_idx+1
        mem_accs.append(memory_loss)
    memory_loss = np.mean(mem_accs)
    memory_std = np.std(mem_accs)

    return memory_loss, memory_std

#-----------------------------------------------------------------------------------------------------------------------------------
# for evaluating performance on the meta-training data 

def get_meta_train_memory_accuracy(model, m_idx, data, target, device):
    meta_train_memory_accuracy = 0

    cur_mem_preds = torch.LongTensor().cpu()
    cur_mem_targets = torch.LongTensor().cpu()
  
    model.cpu()
    memory_data = torch.Tensor()
    memory_targets = torch.Tensor()
    for b_idx in range(0, samples_per_stream*num_tasks, samples_per_stream):
        memory_data = torch.cat((memory_data, data[m_idx][b_idx:b_idx+samples_per_task]), axis=0)
        memory_targets = torch.cat((memory_targets, target[m_idx][b_idx:b_idx+samples_per_task].float()), axis=0)
    memory_data = memory_data.cpu()
    memory_targets = memory_targets.cpu()
    memory_data = Variable(memory_data)
    #check memory
    memory_targets = memory_targets.long()
    mem_output = model(memory_data, memory_data.shape[0], use_dropout=False)
    mem_output = torch.reshape(mem_output[:, :num_hyper_train_tasks], (mem_output.size()[0], num_hyper_train_tasks))
    mem_preds = torch.argmax(mem_output, dim=1)
    
    cur_mem_preds = torch.cat((cur_mem_preds, mem_preds))
    cur_mem_targets = torch.cat((cur_mem_targets, memory_targets))
    model.cpu()

    
    meta_train_memory_task_accuracies = fetch_individual_task_losses(cur_mem_targets, cur_mem_preds, samples_per_task)
    
    meta_train_memory_accuracy = ((cur_mem_targets == cur_mem_preds).sum().item()/(memory_data.shape[0]))*100.
    # mem_accs.append(meta_test_memory_accuracy)
    
    # meta_test_memory_accuracy= np.mean(mem_accs)
    # meta_test_memory_acc_std = np.std(mem_accs)
    # meta_test_memory_task_accuracies = np.mean(model_task_accuracies, axis=0)

    # print("-----------------------------------------")
    # print("meta TEST memory accuracy: ", meta_test_memory_accuracy, "%")
    # print("-----------------------------------------")

    return meta_train_memory_accuracy, meta_train_memory_task_accuracies


#---------------------------------------------------------------------------------------------------------------------------------
# HELPER FUNCTION 
def fetch_individual_task_losses(preds, targets, task_size):
    global num_tasks 
    accs = []
    for i in range(num_tasks):
        start = i*task_size
        end = start+task_size
        accs.append(((targets[start:end] == preds[start:end]).sum().item()/(task_size))*100.)
    return accs 

#-----------------------------------------------------------------------------------------------------------------------------------
# for evaluating performance on a held out dataset (held out classes OMNIGLOT)
def get_meta_test_memory_accuracy(model, m_idx, data, target, device):
    meta_test_memory_accuracy = 0

    cur_mem_preds = torch.LongTensor().cpu()
    cur_mem_targets = torch.LongTensor().cpu()
    model.cpu()
    memory_data = torch.Tensor()
    memory_targets = torch.Tensor()
    for b_idx in range(0,samples_per_stream*num_tasks, samples_per_stream):
        memory_data = torch.cat((memory_data, data[m_idx][b_idx:b_idx+samples_per_task]), axis=0)
        memory_targets = torch.cat((memory_targets, target[m_idx][b_idx:b_idx+samples_per_task].float()), axis=0)
    memory_data = memory_data.cpu()
    memory_targets = memory_targets.cpu()
    memory_data = Variable(memory_data)
    #check memory
    memory_targets = memory_targets.long()
    mem_output = model(memory_data, memory_data.shape[0], use_dropout=False)
    mem_output = torch.reshape(mem_output[:, :num_tasks], (mem_output.size()[0], num_tasks))
    mem_preds = torch.argmax(mem_output, dim=1)
    
    cur_mem_preds = torch.cat((cur_mem_preds, mem_preds))
    cur_mem_targets = torch.cat((cur_mem_targets, memory_targets))
    
    model.cpu()
    
    meta_test_memory_task_accuracies = fetch_individual_task_losses(cur_mem_targets, cur_mem_preds, samples_per_task)
    
    meta_test_memory_accuracy = ((cur_mem_targets == cur_mem_preds).sum().item()/(memory_data.shape[0]))*100.
    # mem_accs.append(meta_test_memory_accuracy)
    
    # meta_test_memory_accuracy= np.mean(mem_accs)
    # meta_test_memory_acc_std = np.std(mem_accs)
    # meta_test_memory_task_accuracies = np.mean(model_task_accuracies, axis=0)

    # print("-----------------------------------------")
    # print("meta TEST memory accuracy: ", meta_test_memory_accuracy, "%")
    # print("-----------------------------------------")

    return meta_test_memory_accuracy, meta_test_memory_task_accuracies

#-----------------------------------------------------------------------------------------------------------------------------------
# for evaluating performance on a held out dataset (MNIST)
def get_meta_eval_memory_accuracy(model, m_idx, data, target, device, samples_per_task, samples_per_stream):
    meta_train_memory_accuracy = 0


    mem_accs = []
    model_task_accuracies = []
    # for m_idx in range(num_models):
    cur_mem_preds = torch.LongTensor().cpu()
    cur_mem_targets = torch.LongTensor().cpu()
    #for eval_mem_idx in range(10):
    model.cpu()

    memory_data = torch.Tensor()
    memory_targets = torch.Tensor()
    for b_idx in range(0,samples_per_stream*num_tasks, samples_per_stream):
        memory_data = torch.cat((memory_data, data[0][b_idx:b_idx+samples_per_task]), axis=0)
        memory_targets = torch.cat((memory_targets, target[0][b_idx:b_idx+samples_per_task].float()), axis=0)
    memory_data = memory_data.cpu()
    memory_targets = memory_targets.cpu()
    memory_data = Variable(memory_data)
    #check memory
    memory_targets = memory_targets.long()
    mem_output = model(memory_data, memory_data.shape[0], use_dropout=False)
    mem_output = torch.reshape(mem_output[:, :num_tasks], (mem_output.size()[0], num_tasks))
    mem_preds = torch.argmax(mem_output, dim=1)
    
    cur_mem_preds = torch.cat((cur_mem_preds, mem_preds))
    cur_mem_targets = torch.cat((cur_mem_targets, memory_targets))
    
    meta_eval_memory_task_accuracies = fetch_individual_task_losses(cur_mem_targets, cur_mem_preds, samples_per_task)
    #model_task_accuracies.append(np.array(meta_eval_memory_task_accuracies).reshape(len(meta_eval_memory_task_accuracies), num_tasks))
    model.cpu()
    meta_eval_memory_accuracy = ((cur_mem_targets == cur_mem_preds).sum().item()/(memory_data.shape[0]))*100.
    #mem_accs.append(meta_train_memory_accuracy)

    # meta_eval_memory_accuracy= np.mean(mem_accs)
    # meta_eval_memory_acc_std = np.std(mem_accs)
    # meta_eval_memory_task_accuracies = np.mean(model_task_accuracies, axis=0)

    # print("-----------------------------------------")
    # print("meta EVAL memory accuracy: ", meta_eval_memory_accuracy, "%")
    # print("-----------------------------------------")

    return meta_eval_memory_accuracy,  meta_eval_memory_task_accuracies

#-----------------------------------------------------------------------------------------------------------------------------------
# for evaluating performance on the meta-training data 
def get_meta_train_generalization_accuracy(model, m_idx, data, target, device):
    meta_train_generalization_accuracy = 0

    

    cur_mem_preds = torch.LongTensor().cpu()
    cur_mem_targets = torch.LongTensor().cpu()
    model.cpu()
    memory_data = torch.Tensor()
    memory_targets = torch.Tensor()
    for b_idx in range(0,samples_per_stream*num_tasks, samples_per_stream):
        
        start = b_idx+samples_per_task
        end = b_idx+25
        #print(start, end)
        memory_data = torch.cat((memory_data, data[m_idx][start:end]), axis=0)
        memory_targets = torch.cat((memory_targets, target[m_idx][start:end].float()), axis=0)
    memory_data = memory_data.cpu()
    memory_targets = memory_targets.cpu()
    memory_data = Variable(memory_data)
    #check memory
    memory_targets = memory_targets.long()
    mem_output = model(memory_data, memory_data.shape[0], use_dropout=False)
    mem_output = torch.reshape(mem_output[:, :num_hyper_train_tasks], (mem_output.size()[0], num_hyper_train_tasks))
    mem_preds = torch.argmax(mem_output, dim=1)
    
    cur_mem_preds = torch.cat((cur_mem_preds, mem_preds))
    cur_mem_targets = torch.cat((cur_mem_targets, memory_targets))
    
    meta_train_generalization_task_accuracies = fetch_individual_task_losses(cur_mem_targets, cur_mem_preds, 25-samples_per_task)
    model.cpu()

    meta_train_generalization_accuracy = ((cur_mem_targets == cur_mem_preds).sum().item()/(memory_data.shape[0]))*100.

    # print("-----------------------------------------")
    # print("meta TEST generalization accuracy: ", meta_test_generalization_accuracy, "%")
    # print("-----------------------------------------")

    return meta_train_generalization_accuracy,  meta_train_generalization_task_accuracies


#-----------------------------------------------------------------------------------------------------------------------------------
# for testing on a held out dataset (held out classes from OMNIGLOT)
def get_meta_test_generalization_accuracy(model, m_idx, data, target, device):
    meta_train_generalization_accuracy = 0

    

    cur_mem_preds = torch.LongTensor().cpu()
    cur_mem_targets = torch.LongTensor().cpu()
    samples_per_task = 5
    model.cpu()
    memory_data = torch.Tensor()
    memory_targets = torch.Tensor()
    for b_idx in range(0,samples_per_stream*num_tasks, samples_per_stream):
        
        start = b_idx+samples_per_task
        end = b_idx+samples_per_stream
        #print(start, end)
        memory_data = torch.cat((memory_data, data[m_idx][start:end]), axis=0)
        memory_targets = torch.cat((memory_targets, target[m_idx][start:end].float()), axis=0)
    memory_data = memory_data.cpu()
    memory_targets = memory_targets.cpu()
    memory_data = Variable(memory_data)
    #check memory
    memory_targets = memory_targets.long()
    mem_output = model(memory_data, memory_data.shape[0], use_dropout=False)
    mem_output = torch.reshape(mem_output[:, :num_tasks], (mem_output.size()[0], num_tasks))
    mem_preds = torch.argmax(mem_output, dim=1)
    
    cur_mem_preds = torch.cat((cur_mem_preds, mem_preds))
    cur_mem_targets = torch.cat((cur_mem_targets, memory_targets))
    
    meta_test_generalization_task_accuracies = fetch_individual_task_losses(cur_mem_targets, cur_mem_preds, samples_per_stream-samples_per_task)
    model.cpu()

    meta_test_generalization_accuracy = ((cur_mem_targets == cur_mem_preds).sum().item()/(memory_data.shape[0]))*100.

    # print("-----------------------------------------")
    # print("meta TEST generalization accuracy: ", meta_test_generalization_accuracy, "%")
    # print("-----------------------------------------")

    return meta_test_generalization_accuracy,  meta_test_generalization_task_accuracies


#-----------------------------------------------------------------------------------------------------------------------------------
# for testing on a held out dataset (MNIST)
def get_meta_eval_generalization_accuracy(model, m_idx, data, target, device,  eval_samples_per_task, eval_samples_per_stream):
    meta_train_generalization_accuracy = 0

    mem_accs = []
    model_task_accuracies = []
   

    cur_mem_preds = torch.LongTensor().cpu()
    cur_mem_targets = torch.LongTensor().cpu()
    samples_per_task = eval_samples_per_task
    samples_per_stream = eval_samples_per_stream
    model.cpu()

    memory_data = torch.Tensor()
    memory_targets = torch.Tensor()
    for b_idx in range(0,samples_per_stream*num_tasks, samples_per_stream):
        
        start = b_idx+samples_per_task
        end = b_idx+samples_per_stream
        #print(start, end)
        memory_data = torch.cat((memory_data, data[0][start:end]), axis=0)
        memory_targets = torch.cat((memory_targets, target[0][start:end].float()), axis=0)
    memory_data = memory_data.cpu()
    memory_targets = memory_targets.cpu()
    memory_data = Variable(memory_data)
    #check memory
    memory_targets = memory_targets.long()
    mem_output = model(memory_data, memory_data.shape[0], use_dropout=False)
    mem_output = torch.reshape(mem_output[:, :num_tasks], (mem_output.size()[0], num_tasks))
    mem_preds = torch.argmax(mem_output, dim=1)
    
    cur_mem_preds = torch.cat((cur_mem_preds, mem_preds))
    cur_mem_targets = torch.cat((cur_mem_targets, memory_targets))
    model.cpu()
    meta_eval_generalization_task_accuracies = fetch_individual_task_losses(cur_mem_targets, cur_mem_preds, samples_per_stream-samples_per_task)
    #model_task_accuracies.append(np.array(meta_eval_generalization_task_accuracies).reshape(len(meta_eval_generalization_task_accuracies), num_tasks))

    meta_eval_generalization_accuracy = ((cur_mem_targets == cur_mem_preds).sum().item()/(memory_data.shape[0]))*100.
    #mem_accs.append(meta_eval_generalization_accuracy)

    # meta_eval_generalization_accuracy = np.mean(mem_accs)
    # meta_eval_genrl_std = np.std(mem_accs)
    # meta_eval_generalization_task_accuracies = np.mean(model_task_accuracies, axis=0)

    # print("-----------------------------------------")
    # print("meta eval generalization accuracy: ", meta_eval_generalization_accuracy, "%")
    # print("-----------------------------------------")

    return meta_eval_generalization_accuracy,  meta_eval_generalization_task_accuracies

#-----------------------------------------------------------------------------------------------------------------------------------
def fetch_hyper_data(data, target, samples_per_task, hyper_batch_end, samples_per_stream, device):
    hyper_data = torch.Tensor()
    hyper_targets = torch.Tensor()
    hyper_indices = np.random.randint(0, data.shape[0], 25)
    hyper_data = data[hyper_indices]
    hyper_targets = target[hyper_indices].float()
    # for b_idx in range(0, data.shape[0], samples_per_stream):
    #     init = b_idx
    #     start = b_idx #+ samples_per_task 
    #     end = init + hyper_batch_end
    #     hyper_indices = np.random.randint(0, hyper_batch_end, 5)
    #     hyper_data = torch.cat((hyper_data, data[start:end][hyper_indices]), axis=0)
    #     hyper_targets = torch.cat((hyper_targets, target[start:end][hyper_indices].float()), axis=0)
    return hyper_data, hyper_targets 


def fetch_X_N_hyper_data(task_idx, data, target, samples_per_task, hyper_batch_end, samples_per_stream, device):
    hyper_data = torch.Tensor()
    hyper_targets = torch.Tensor()
    for b_idx in range(0, (task_idx+1)*samples_per_stream, samples_per_stream):
        init = b_idx
        start = b_idx #+ samples_per_task 
        end = init + samples_per_task #hyper_batch_end
        hyper_indices = np.random.randint(0, samples_per_task, 3)
        hyper_data = torch.cat((hyper_data, data[start:end][hyper_indices]), axis=0)
        hyper_targets = torch.cat((hyper_targets, target[start:end][hyper_indices].float()), axis=0)
    return hyper_data, hyper_targets 

#-----------------------------------------------------------------------------------------------------------------------------------
def fetch_augmented_data(data, Data_Augmenter):
    augmented_images = torch.zeros_like(data)
    for idx, img, in enumerate(data): 
        augmented_image = Data_Augmenter(img)
        augmented_images[idx] = augmented_image
    return augmented_images

#-----------------------------------------------------------------------------------------------------------------------------------
all_losses = []
all_stds = []
def train(args, models, device, data_loader, epoch, dims, n_channels, eval_data_loader, omniglot_test_data_loader, Data_Augmenter):
    global loss_scaler
    global my_batch_size
    samples_per_task = my_batch_size
    global samples_per_stream
    global num_tasks 
    global num_img_tasks
    global test_tasks 
    global k_steps 
    #model.train()
    losses = []
    #for batch_idx, (data, target) in enumerate(train_loader):
    all_memory_losses = []
    all_memory_stds = []
    batch_idx = 0


    

    all_avg_memory_losses = []
    all_avg_memory_stds = []
    META_TRAIN_MEMORY_ACCURACIES = []
    META_TRAIN_GENERALIZATION_ACCURACIES = []
    META_TRAIN_MEMORY_STD = []
    META_TRAIN_GENERALIZATION_STD = []

    ALL_META_TRAIN_MEMORY_TASK_ACCURACIES = []
    ALL_META_TRAIN_GENERALIZATION_TASK_ACCURACIES = []

    META_TEST_MEMORY_ACCURACIES = []
    META_TEST_GENERALIZATION_ACCURACIES = [] 
    META_TEST_MEMORY_STD = []
    META_TEST_GENERALIZATION_STD = [] 

    ALL_META_TEST_MEMORY_TASK_ACCURACIES = []
    ALL_META_TEST_GENERALIZATION_TASK_ACCURACIES = []

    META_EVAL_MEMORY_ACCURACIES = []
    META_EVAL_GENERALIZATION_ACCURACIES = [] 
    META_EVAL_MEMORY_STD = []
    META_EVAL_GENERALIZATION_STD = []

    ALL_META_EVAL_MEMORY_TASK_ACCURACIES = []
    ALL_META_EVAL_GENERALIZATION_TASK_ACCURACIES = []

    minimum_loss = 100
    prior_lr = .1

    
    prior = copy.deepcopy(models["m1"]["model"]).cpu()
    prior_opt = torch.optim.Adam(prior.VECS_layers.parameters(),  lr=vecs_lr)
    
    
    
    iid_HLT_data = torch.Tensor()
    for batch in data_loader:
        #print("batch_idx: ", batch_idx)
        
        """contains 3 streams of data (num_task sequences) already, just have to break it up"""
        samples_per_stream = 19*samples_per_stream_multiplier
        samples_per_task = 5
        my_batch_size = samples_per_task
      
        data, target = batch["train"]

       
        
        data = data.repeat_interleave(samples_per_stream_multiplier, dim=1)
        target = target.repeat_interleave(samples_per_stream_multiplier, dim=1)

        data = data.repeat(1,1,3,1,1)
        
     
        
        #put fashion mnist data on 1 stream 
        (fmnist_data, fmnist_targets) = get_Fashion_MNIST_Hyper_Stream(10, 19*samples_per_stream_multiplier)
        fmnist_data = fmnist_data.repeat(1,1,3,1,1)
        
        fmnist_targets += 100+num_hyper_train_CL_tasks
        

        (cifar_data, cifar_targets) = get_CIFAR100_Hyper_Stream(100, 19*samples_per_stream_multiplier)
        cifar_targets += num_hyper_train_CL_tasks

        for stream in range(k_streams):
            rand_labels = np.arange(num_hyper_train_tasks)
            random.shuffle(rand_labels)
            
            for t in range(num_hyper_train_CL_tasks):
                prb = np.random.random()
                
                if prb < .1:
                    rand_t_idx = np.random.randint(10)
                    t_start = (19*samples_per_stream_multiplier)*t
                    t_end = t_start + (19*samples_per_stream_multiplier)
                    rand_t_start = (19*samples_per_stream_multiplier)*rand_t_idx
                    rand_t_end = rand_t_start + (19*samples_per_stream_multiplier)
                    # print("fmnist", stream, t, rand_t_idx)
                    # # print(fmnist_targets[rand_t_start:rand_t_end].shape)
                    # # print(data[stream, t_start:t_end].shape)
                    random_label = torch.Tensor(rand_labels[int(fmnist_targets[rand_t_start])].repeat(len(fmnist_targets[rand_t_start:rand_t_end]))).long()
                    # print("------------------")
                    data[stream, t_start:t_end] =  fmnist_data[0][rand_t_start:rand_t_end]
                    target[stream, t_start:t_end] = random_label #fmnist_targets[rand_t_start:rand_t_end].long()
                elif prb >= .6:
                    rand_t_idx = np.random.randint(100)
                    t_start = (19*samples_per_stream_multiplier)*t
                    t_end = t_start + (19*samples_per_stream_multiplier)
                    rand_t_start = (19*samples_per_stream_multiplier)*rand_t_idx
                    rand_t_end = rand_t_start + (19*samples_per_stream_multiplier)
                    # print("cifar", stream, t, rand_t_idx)
                    # # print(cifar_data[rand_t_start:rand_t_end].shape)
                    # # print(cifar_targets[rand_t_start:rand_t_end].shape)
                    # print("------------------")
                    data[stream, t_start:t_end] =  cifar_data[rand_t_start:rand_t_end]
                   
                    random_label = torch.Tensor(rand_labels[int(cifar_targets[rand_t_start])].repeat(len(cifar_targets[rand_t_start:rand_t_end]))).long()
                    
                    target[stream, t_start:t_end] = random_label #cifar_targets[rand_t_start:rand_t_end].long()
            
            # many-to-one classification, i.e. map group of classes to single labels 
            # #the idea is to train the system to retain memory of many overlapping classes 
            # making the problem more difficult, which may improve deployment performance on the 
            # simpler task of performing class incremental learning when the mapping is one-to-one  
            # print(target[stream][::19*samples_per_stream_multiplier])
            # target = target%CL_task_sequence_length
            # print(target[stream][::19*samples_per_stream_multiplier])
           

        

        
        
    
        # augmented_data = Data_Augmenter(data[0][0])
        # save_image(data[0][0], "actaul_image.png")
        # save_image(augmented_data, "augmented_image.png")
        # save_image(augmented_prime, "augmented_prime_image.png")
        # data, target = data.to(device), target.to(device)
        # augmented_data = augmented_data.to(device)
        # print(augmented_data.shape)
        # sys.exit()

        # augmented_data= augmented_data.to(device)

        # #data = data.to(device)
        # #target = one_hot_embedding(target, num_tasks)
        
        # print(data.shape)
        # print(augmented_data.shape)
        # sys.exit()

        
        

        #print('Train inputs shape: {0}'.format(data[0][:50].shape))    # (16, 25, 1, 28, 28)
        # print('Train targets shape: {0}'.format(target.shape))  # (16, 25)
    
        if batch_idx > 0: #then load the saved model
            models = reset_models(models, dims, n_channels, device, batch_idx)
           

        #--inner loop ----------------------------------------------------------------------------------------------
        HLT_memory_losses = []
        HLT_memory_stds = []
        

        dan_grads = {}
        grad_steps = 1
        
        
        vecs_grads = {}
        
        for i in range(CL_task_sequence_length):
            print("Epoch:",batch_idx,   " /  Task:", i)
            
            
            data_streams = {}
            for m_idx in range(num_models):
                data_streams["m"+str(m_idx+1)] = {}
       
            
            if batch_idx > -1:    
                for step in range(k_steps):
                    
                    batch_start = i*samples_per_stream  
                    
                    for m_idx in range(num_models):
                        data_streams["m"+str(m_idx+1)]["curdata"] = torch.Tensor()
                        data_streams["m"+str(m_idx+1)]["cur_target"] = torch.Tensor()
                   
                    
                    dan_grads = {}
                    
                    #put the Synapses into their new state induced by the current non-iid data (and the DANs, which are assumed as given)
                    for model_idx, model_key in enumerate(models.keys()):

                        models["m"+str(model_idx+1)]["vecs_opt"].zero_grad()
                        models["m"+str(model_idx+1)]["dan_opt"].zero_grad()
                        models["m"+str(model_idx+1)]["model"].to(device)
                
                        #synapse_indices = np.random.randint(0,data[model_idx].shape[0], num_hyper_train_samples_per_task)
                        synapse_indices = np.random.randint(batch_start , batch_start+samples_per_stream, num_hyper_train_samples_per_task)
                        data_streams["m"+str(model_idx+1)]["curdata"] = data[model_idx][synapse_indices].reshape((num_hyper_train_samples_per_task,3,32,32))
                        data_streams["m"+str(model_idx+1)]["cur_target"] = target[model_idx][synapse_indices].reshape((num_hyper_train_samples_per_task)).float()
                        
                            
                        #sample_indices = np.random.randint(0, data_streams["m"+str(model_idx+1)]["curdata"].shape[0], num_hyper_train_samples_per_task)
                        curdata = data_streams["m"+str(model_idx+1)]["curdata"]#[sample_indices]
                        cur_targets = data_streams["m"+str(model_idx+1)]["cur_target"]#[sample_indices]

                        # augment the training data for synapses
                        synapse_data = fetch_augmented_data(data_streams["m"+str(model_idx+1)]["curdata"], Data_Augmenter)
                        
                        synapse_data = torch.cat((synapse_data, data_streams["m"+str(model_idx+1)]["curdata"]))
                        synapse_targets = torch.cat((data_streams["m"+str(model_idx+1)]["cur_target"], data_streams["m"+str(model_idx+1)]["cur_target"]))
                    
                        sample_idx = np.random.randint(0, synapse_data.shape[0], 3)
                        synapse_data = synapse_data[sample_idx]
                        synapse_targets = synapse_targets[sample_idx]

                        synapse_data, synapse_targets = synapse_data.to(device), synapse_targets.to(device)
                        synapse_data = Variable(synapse_data, requires_grad=True)
                        synapse_targets = Variable(synapse_targets, requires_grad=True)
                        cur_target = synapse_targets.long()
                        #cur_target = one_hot_embedding(cur_target, num_hyper_train_tasks).to(device)



                        # COMPUTE GRADIENTS FOR VECS USING NON-IID DATA, AND TAKE A STEP
                        models["m"+str(model_idx+1)]["output"] = models["m"+str(model_idx+1)]["model"](synapse_data, synapse_data.shape[0], train_step=True)
                        models["m"+str(model_idx+1)]["output"] = torch.reshape(models["m"+str(model_idx+1)]["output"], (models["m"+str(model_idx+1)]["output"].size()[0],num_hyper_train_tasks))
                                        
                        
                        models["m"+str(model_idx+1)]["loss"] = F.nll_loss(models["m"+str(model_idx+1)]["output"], cur_target)  #nn.CrossEntropyLoss()(models["m"+str(model_idx+1)]["output"], cur_target)
                        models["m"+str(model_idx+1)]["loss"].backward()                     
                        models["m"+str(model_idx+1)]["vecs_opt"].step()
                    

                    for model_idx, model_key in enumerate(models.keys()):
                        models["m"+str(model_idx+1)]["hyper_loss"] = 0
                        models["m"+str(model_idx+1)]["dan_opt"].zero_grad()
                        models["m"+str(model_idx+1)]["vecs_opt"].zero_grad()

                        for hyper_step in range(num_hyper_steps):
                            hyper_data, hyper_targets = fetch_hyper_data(data[model_idx], target[model_idx], samples_per_task, hyper_batch_end, samples_per_stream, device) 
                            hyper_indices = np.random.randint(0, hyper_data.shape[0], hyper_batch_size)
                            augmented_hyper_data = fetch_augmented_data(hyper_data[hyper_indices], Data_Augmenter)
                            augmented_hyper_data = torch.cat((augmented_hyper_data, hyper_data[hyper_indices]))
                            data_streams["m"+str(model_idx+1)]["sampled_HLT_data"] = Variable(augmented_hyper_data, requires_grad=True).to(device)
                            hyper_targets = torch.cat((hyper_targets[hyper_indices], hyper_targets[hyper_indices]))
                            
                           
                            
                            data_streams["m"+str(model_idx+1)]["sampled_HLT_targets"] = Variable(hyper_targets.float(), requires_grad=True)
                            raw_hyper_targets = data_streams["m"+str(model_idx+1)]["sampled_HLT_targets"].long()
                            raw_hyper_targets = raw_hyper_targets.to(device)
                            #hyper_targets = one_hot_embedding(raw_hyper_targets, num_hyper_train_tasks).to(device)
                        
                        
                            models["m"+str(model_idx+1)]["hlt_output"]  = models["m"+str(model_idx+1)]["model"](data_streams["m"+str(model_idx+1)]["sampled_HLT_data"], hyper_batch_size*2, DAN_dropout=False, train_step=True)
                            models["m"+str(model_idx+1)]["hlt_output"] = torch.reshape(models["m"+str(model_idx+1)]["hlt_output"], (models["m"+str(model_idx+1)]["hlt_output"].size()[0],num_hyper_train_tasks))
                        
                            #gradient descent using this gradient 
                            models["m"+str(model_idx+1)]["hyper_loss"] += F.nll_loss(models["m"+str(model_idx+1)]["hlt_output"], raw_hyper_targets) #nn.CrossEntropyLoss()(models["m"+str(model_idx+1)]["hlt_output"], data_streams["m"+str(model_idx+1)]["sampled_HLT_targets"].long())            
                            #+ models["m"+str(model_idx+1)]["XN_loss"]#+ param_mse
                            models["m"+str(model_idx+1)]["hyper_loss"].backward(retain_graph=True)
                       
                      

                        if model_idx == 0:
                            for name, param in models["m"+str(model_idx+1)]["model"].DAN_layers.named_parameters():
                                dan_grads[name] = param.grad.clone()
                        
                        #     # if i == CL_task_sequence_length-1 and step == k_steps-1:
                        #     #     for name, param in models["m"+str(model_idx+1)]["model"].VECS_layers.named_parameters():
                        #     #         vecs_grads[name] = param.grad.clone().detach().cpu()
                        
                    
                        else:
                            for name, param in models["m"+str(model_idx+1)]["model"].DAN_layers.named_parameters():
                                dan_grads[name] += param.grad.clone()

                            # if i == CL_task_sequence_length-1 and step == k_steps-1:
                            #     for name, param in models["m"+str(model_idx+1)]["model"].VECS_layers.named_parameters():
                            #         vecs_grads[name] += param.grad.clone().detach().cpu()
                        
                    # #---------------------------------------------------------------------------------------------------------------------------------------------
                    
                    # #NOW AVERAGE THE DAN GRADS, AND UPDATE THE DANS IN ALL MODELS. 
                    for model_idx, model_key in enumerate(models.keys()):
    
                        models["m"+str(model_idx+1)]["dan_opt"].zero_grad()
                        for name, param in models["m"+str(model_idx+1)]["model"].DAN_layers.named_parameters():
                            param.grad = dan_grads[name]
                        models["m"+str(model_idx+1)]["dan_opt"].step()
        
                        

            #meta_train_memory_loss, meta_train_memory_std = get_meta_train_memory_loss(models, data, target, i, device)
            HLT_memory_losses.append(0)
            HLT_memory_stds.append(0)

       

        # sum_optimal = {}
        # for model_idx in range(num_models):
            
        #     models["m"+str(model_idx+1)]["model"].to("cpu")
        #     if model_idx == 0:
        #         for (name, param), (pname, pparam) in zip(models["m"+str(model_idx+1)]["model"].VECS_layers.named_parameters(), prior.VECS_layers.named_parameters()):
        #             sum_optimal[name] = (param.data.clone() - pparam.data)/num_models 
        #     else:
        #         for (name, param), (pname, pparam) in zip(models["m"+str(model_idx+1)]["model"].VECS_layers.named_parameters(), prior.VECS_layers.named_parameters()):
        #             sum_optimal[name] += (param.data.clone() - pparam.data)/num_models
    
        
        # for name, param in prior.VECS_layers.named_parameters():
        #     param.data += sum_optimal[name]
        
      

        # prior.to(device)
        for model_idx, model_key in enumerate(models.keys()):
            
            # models["m"+str(model_idx+1)]["model"].VECS_layers.load_state_dict(prior.VECS_layers.state_dict())
            torch.save(models["m"+str(model_idx+1)]["model"].state_dict(), "noniid_hyper_learning16_ultra2_m"+str(model_idx+1)+".pt")
        # prior.to("cpu")   
            #--------------------------------------------------------------------------------------------------
        

        all_memory_losses.append(np.mean(HLT_memory_losses))
        all_memory_stds.append(np.mean(HLT_memory_stds))
       
        #  visualize the memory loss -----------------------------------------------------------------------
      
        check_idx = 1
        lml = len(all_memory_losses)   
        
        print("Train epoch: ",batch_idx," / LOSS: ", np.mean(all_memory_losses[lml-check_idx:lml]))
        print("------------------------------------------------------------------------")
        if batch_idx % check_idx == 0:
           
            #-------- NEW META-TRAIN----------------------------------------------------------------------
            meta_train_memory_accuracy, meta_train_memory_std, meta_train_generalization_accuracy, meta_train_generalization_std, meta_train_memory_task_accuracies, meta_train_generalization_task_accuracies = test_train_omniglot(device, data, target, dims, n_channels, samples_per_task, hyper_batch_size, samples_per_stream, num_tasks, k_steps, Data_Augmenter)
            
            
            META_TRAIN_MEMORY_ACCURACIES.append(meta_train_memory_accuracy)
            META_TRAIN_GENERALIZATION_ACCURACIES.append(meta_train_generalization_accuracy)

            META_TRAIN_MEMORY_STD.append(meta_train_memory_std)
            META_TRAIN_GENERALIZATION_STD.append(meta_train_generalization_std)

            ALL_META_TRAIN_MEMORY_TASK_ACCURACIES.append(meta_train_memory_task_accuracies)
            ALL_META_TRAIN_GENERALIZATION_TASK_ACCURACIES.append(meta_train_generalization_task_accuracies)

            with open("noniid_hyper_learning16_ultra2_ALL_META_TRAIN_MEMORY_TASK_ACCURACIES.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(ALL_META_TRAIN_MEMORY_TASK_ACCURACIES)

            with open("noniid_hyper_learning16_ultra2_ALL_META_TRAIN_GENERALIZATION_TASK_ACCURACIES.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(ALL_META_TRAIN_GENERALIZATION_TASK_ACCURACIES)
            #---------END NEW META-TRAIN ----------------------------------------------------------------


            # meta_test_memory_mean, meta_test_memory_std, meta_test_generalization_mean, meta_test_generalization_std, memory_task_accuracies, genrl_task_accuracies

            meta_test_memory_accuracy, meta_test_memory_std, meta_test_generalization_accuracy, meta_test_generalization_std, meta_test_memory_task_accuracies, meta_test_generalization_task_accuracies = test_ood_omniglot(device, omniglot_test_data_loader, dims, n_channels, samples_per_task, hyper_batch_size, num_tasks, k_steps, Data_Augmenter)
            
            
            META_TEST_MEMORY_ACCURACIES.append(meta_test_memory_accuracy)
            META_TEST_GENERALIZATION_ACCURACIES.append(meta_test_generalization_accuracy)

            META_TEST_MEMORY_STD.append(meta_test_memory_std)
            META_TEST_GENERALIZATION_STD.append(meta_test_generalization_std)

            ALL_META_TEST_MEMORY_TASK_ACCURACIES.append(meta_test_memory_task_accuracies)
            ALL_META_TEST_GENERALIZATION_TASK_ACCURACIES.append(meta_test_generalization_task_accuracies)

            

            with open("noniid_hyper_learning16_ultra2_ALL_META_TEST_MEMORY_TASK_ACCURACIES.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(ALL_META_TEST_MEMORY_TASK_ACCURACIES)

            with open("noniid_hyper_learning16_ultra2_ALL_META_TEST_GENERALIZATION_TASK_ACCURACIES.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(ALL_META_TEST_GENERALIZATION_TASK_ACCURACIES)
            
            all_avg_memory_losses.append(round(np.mean(all_memory_losses[lml-check_idx:lml]), 4))
            all_avg_memory_stds.append(round(np.mean(all_memory_stds[lml-check_idx:lml]), 4))
            fig, ax1 = plt.subplots()
            ax1.set_title("META-TRAIN MEMORY LOSS", fontsize=12, color="black")
            ax1.set_ylabel('Meta-Train Memory Loss', color="red")  # we already handled the x-label with ax1
            ax1.tick_params(axis='y', labelcolor="red")
            ax1.set_xlabel('Eval Iteration', color="black")
          
            x_plotLength = np.arange(0,len(all_memory_losses), check_idx)
        
            line1, = ax1.plot(x_plotLength, all_avg_memory_losses, c="red", label="Meta-Train")

           
           
            ax2 = ax1.twinx() 
            # line2, eval_memory_losses, eval_memory_losses_std, meta_eval_memory_accuracy, meta_eval_memory_std, meta_eval_generalization_accuracy, meta_eval_generalization_std, meta_eval_memory_task_accuracies, meta_eval_generalization_task_accuracies
            line2, eval_memory_losses, eval_memory_stds, meta_eval_memory_accuracy, meta_eval_memory_std, meta_eval_generalization_accuracy, meta_eval_generalization_std, meta_eval_memory_task_accuracies, meta_eval_generalization_task_accuracies = test( device, eval_data_loader, dims, n_channels, samples_per_task, test_tasks, k_steps, ax2, x_plotLength, Data_Augmenter)
            eval_memory_loss = eval_memory_losses[-1]
            META_EVAL_MEMORY_ACCURACIES.append(meta_eval_memory_accuracy)
            META_EVAL_GENERALIZATION_ACCURACIES.append(meta_eval_generalization_accuracy)

            ALL_META_EVAL_MEMORY_TASK_ACCURACIES.append(meta_eval_memory_task_accuracies)
            ALL_META_EVAL_GENERALIZATION_TASK_ACCURACIES.append(meta_eval_generalization_task_accuracies)

            META_EVAL_MEMORY_STD.append(meta_eval_memory_std)
            META_EVAL_GENERALIZATION_STD.append(meta_eval_generalization_std)

            with open("noniid_hyper_learning16_ultra2_ALL_META_EVAL_MEMORY_TASK_ACCURACIES.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(ALL_META_EVAL_MEMORY_TASK_ACCURACIES)

            with open("noniid_hyper_learning16_ultra2_ALL_META_EVAL_GENERALIZATION_TASK_ACCURACIES.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(ALL_META_EVAL_GENERALIZATION_TASK_ACCURACIES)

            np.savetxt("noniid_hyper_learning16_ultra2_results.csv", np.column_stack((all_avg_memory_losses, META_TRAIN_MEMORY_ACCURACIES, META_TRAIN_GENERALIZATION_ACCURACIES, META_TEST_MEMORY_ACCURACIES, META_TEST_GENERALIZATION_ACCURACIES, eval_memory_losses, META_EVAL_MEMORY_ACCURACIES, META_EVAL_GENERALIZATION_ACCURACIES)), delimiter=",", fmt='%s')
            np.savetxt("noniid_hyper_learning16_ultra2_results_stds.csv", np.column_stack((all_avg_memory_stds, META_TRAIN_MEMORY_STD, META_TRAIN_GENERALIZATION_STD, META_TEST_MEMORY_STD, META_TEST_GENERALIZATION_STD, eval_memory_stds, META_EVAL_MEMORY_STD, META_EVAL_GENERALIZATION_STD)), delimiter=",", fmt='%s')

            # sys.exit()
            print("Eval Memory Loss: ", eval_memory_loss)
            
            ax1.grid(True, color="lightgrey")
            ax1.set_ylim(0)
            ax1.legend(handles=[line1, line2], loc='upper right')
            fig.tight_layout() 
            plt.savefig("noniid_hyper_learning16_ultra2_training_loss.jpg")
            # if (np.mean(all_memory_losses[lml-check_idx:lml])) < .0008:
            #     sys.exit()
            plt.close('all')

            if eval_memory_loss < minimum_loss:
                minimum_loss = eval_memory_loss

             
                torch.save(models["m1"]["model"].state_dict(), "m1_noniid_hyper_learning16_ultra2_optimal_model.pt")

        
        batch_idx += 1


def test_ood_omniglot(device, omniglot_test_data_loader, dims, n_channels, samples_per_task, hyper_batch_size,  num_tasks, k_steps, Data_Augmenter):
    global samples_per_stream
    for batch in omniglot_test_data_loader: 
        """contains k streams of data (num_task sequences)"""
        data, target = batch["train"]

        
        
        data = data.repeat(1,1,3,1,1)
        # data, target = data.to(device), target
        # target = target.to(device)

        mem_accuracy = []
        memory_task_accuracies = []

        samples_per_stream = 19
    
        genrl_accuracy = []
        genrl_task_accuracies = []

        for m_idx in range(3):
            torch.manual_seed(torch.randint(0,400, (1,)).item())
            model = DAN_classifier(dims=dims, n_channels=n_channels)
            phenotype_model = DAN_classifier(dims=dims, n_channels=n_channels)
            phenotype_model.load_state_dict(torch.load("noniid_hyper_learning16_ultra2_m"+str(m_idx+1)+".pt")) 
            model.DAN_layers.load_state_dict(phenotype_model.DAN_layers.state_dict())
            #model.VECS_layers.load_state_dict(phenotype_model.VECS_layers.state_dict())
            model = model.to(device)
            vecs_optimizer = torch.optim.Adam(model.VECS_layers.parameters(),  lr=vecs_lr)
            dan_optimizer = torch.optim.Adam(model.DAN_layers.parameters(),  lr=dan_lr)

            del phenotype_model

            # # use this for iid training ----------------------------------------
            # training_data = torch.Tensor()
            # training_targets = torch.Tensor()
            # for i in range(num_tasks):
            #     batch_start = i*samples_per_stream
            #     training_data = torch.cat((training_data, data[m_idx][batch_start: batch_start+samples_per_task])) 
            #     training_targets = torch.cat((training_targets, target[m_idx][batch_start: batch_start+samples_per_task].float()))
            #  # (end) use this for iid training ----------------------------------------
            
            for i in range(num_tasks):
                
                
                
                for step in range(test_steps):
                    #for non-iid training
                    batch_start = i*samples_per_stream 
                    batch_end = batch_start + samples_per_task
                    synapse_indices = np.random.randint(batch_start, batch_end, samples_per_task)
                    test_curdata = data[m_idx][synapse_indices].view((samples_per_task,3,32,32))
                    test_cur_targets = target[m_idx][synapse_indices].view((samples_per_task))

                    # #for iid training
                    # synapse_indices = np.random.randint(0, len(training_data), samples_per_task)
                    # test_curdata = training_data[synapse_indices].view((samples_per_task,3,32,32))
                    # test_cur_targets = training_targets[synapse_indices].view((samples_per_task))

                    

                    
                    test_cur_targets = test_cur_targets.long()
                    test_cur_targets = torch.cat((test_cur_targets, test_cur_targets))
                
                    vecs_optimizer.zero_grad() 
                    dan_optimizer.zero_grad()


                    augmented_test_data = fetch_augmented_data(test_curdata.cpu(), Data_Augmenter)
                    
                    
                    test_curdata = torch.cat((test_curdata, augmented_test_data))
                    
                    sample_idx = np.random.randint(0, test_curdata.shape[0], 3)
                    test_curdata = test_curdata[sample_idx]
                    test_cur_targets = test_cur_targets[sample_idx]
                    test_curdata = test_curdata.to(device)
                    test_cur_targets = test_cur_targets.to(device)
                    #cur_target = one_hot_embedding(test_cur_targets, num_hyper_train_tasks).to(device)
                    
                    test_output = model(test_curdata, samples_per_task*2, use_dropout=False, train_step=True)
                    
                    
                    test_output = torch.reshape(test_output[:, :], (test_output.size()[0],num_hyper_train_tasks))
                    
                    test_loss = F.nll_loss(test_output, test_cur_targets) 
                    test_loss.backward()
                    vecs_optimizer.step()
                    #dan_optimizer.step()

            meta_test_memory_accuracy, meta_test_memory_task_accuracies = get_meta_test_memory_accuracy(model, m_idx, data, target, device)
            mem_accuracy.append(meta_test_memory_accuracy)
            memory_task_accuracies.append(meta_test_memory_task_accuracies)

            meta_test_generalization_accuracy, meta_test_generalization_task_accuracies = get_meta_test_generalization_accuracy(model, m_idx, data, target, device)
            genrl_accuracy.append(meta_test_generalization_accuracy)
            genrl_task_accuracies.append(meta_test_generalization_task_accuracies)
            model.cpu()
        memory_task_accuracies = np.array(memory_task_accuracies).reshape(len(memory_task_accuracies), num_tasks)
        genrl_task_accuracies = np.array(genrl_task_accuracies).reshape(len(genrl_task_accuracies), num_tasks)

        memory_task_accuracies = np.mean(memory_task_accuracies, axis=0)
        genrl_task_accuracies = np.mean(genrl_task_accuracies, axis=0)
        
        meta_test_memory_mean = np.mean(mem_accuracy)
        meta_test_memory_std = np.std(mem_accuracy)

        meta_test_generalization_mean = np.mean(genrl_accuracy)
        meta_test_generalization_std = np.std(genrl_accuracy)



        return meta_test_memory_mean, meta_test_memory_std, meta_test_generalization_mean, meta_test_generalization_std, memory_task_accuracies, genrl_task_accuracies



##################################################################################################################################
#    META TRAIN TEST 

def test_train_omniglot(device, data, target, dims, n_channels, samples_per_task, hyper_batch_size, samples_per_stream, num_tasks, k_steps, Data_Augmenter):
   
    mem_accuracy = []
    memory_task_accuracies = []

    genrl_accuracy = []
    genrl_task_accuracies = []



    
    for m_idx in range(3):
        torch.manual_seed(torch.randint(0,400, (1,)).item())
        model = DAN_classifier(dims=dims, n_channels=n_channels)
        phenotype_model = DAN_classifier(dims=dims, n_channels=n_channels)
        phenotype_model.load_state_dict(torch.load("noniid_hyper_learning16_ultra2_m"+str(m_idx+1)+".pt")) 
        model.DAN_layers.load_state_dict(phenotype_model.DAN_layers.state_dict())
        #model.VECS_layers.load_state_dict(phenotype_model.VECS_layers.state_dict())
        model = model.to(device)
        vecs_optimizer = torch.optim.Adam(model.VECS_layers.parameters(),  lr=vecs_lr)
        dan_optimizer =  torch.optim.Adam(model.DAN_layers.parameters(),  lr=dan_lr)
        del phenotype_model

        # training_data = torch.Tensor()
        # training_targets = torch.Tensor()
        # for i in range(num_tasks):
        #     batch_start = i*samples_per_stream
        #     training_data = torch.cat((training_data, data[m_idx][batch_start: batch_start+samples_per_task])) 
        #     training_targets = torch.cat((training_targets, target[m_idx][batch_start: batch_start+samples_per_task].float()))

        for i in range(num_tasks):
           
            
            for step in range(test_steps):
                batch_start = i*samples_per_stream 
                batch_end = batch_start + samples_per_task
                synapse_indices = np.random.randint(batch_start,batch_end, samples_per_task)
                test_curdata = data[m_idx][synapse_indices].view((samples_per_task,3,32,32))
                test_cur_targets = target[m_idx][synapse_indices].view((samples_per_task))

                
                
                test_cur_targets = test_cur_targets.long()
                test_cur_targets = torch.cat((test_cur_targets, test_cur_targets))
                
                vecs_optimizer.zero_grad() 
                dan_optimizer.zero_grad()

                augmented_test_data = fetch_augmented_data(test_curdata.cpu(), Data_Augmenter)
            
                test_curdata = torch.cat((test_curdata, augmented_test_data))
                
                
                sample_idx = np.random.randint(0, test_curdata.shape[0], 3)
                test_curdata = test_curdata[sample_idx]
                test_cur_targets = test_cur_targets[sample_idx]
                test_curdata = test_curdata.to(device)
                test_cur_targets = test_cur_targets.to(device)
                #cur_target = one_hot_embedding(test_cur_targets, num_hyper_train_tasks).to(device)
                

                test_output = model(test_curdata, samples_per_task*2, use_dropout=False, train_step=True)
                
                
                test_output = torch.reshape(test_output[:, :num_hyper_train_tasks], (test_output.size()[0],num_hyper_train_tasks))
                
                test_loss = F.nll_loss(test_output, test_cur_targets) 
                test_loss.backward()
                vecs_optimizer.step()
                #dan_optimizer.step()

        meta_train_memory_accuracy, meta_train_memory_task_accuracies = get_meta_train_memory_accuracy(model, m_idx, data, target, device)
        mem_accuracy.append(meta_train_memory_accuracy)
        memory_task_accuracies.append(meta_train_memory_task_accuracies)

        meta_train_generalization_accuracy, meta_train_generalization_task_accuracies = get_meta_train_generalization_accuracy(model, m_idx, data, target, device)
        genrl_accuracy.append(meta_train_generalization_accuracy)
        genrl_task_accuracies.append(meta_train_generalization_task_accuracies)
    model.cpu()
    memory_task_accuracies = np.array(memory_task_accuracies).reshape(len(memory_task_accuracies), num_tasks)
    genrl_task_accuracies = np.array(genrl_task_accuracies).reshape(len(genrl_task_accuracies), num_tasks)

    memory_task_accuracies = np.mean(memory_task_accuracies, axis=0)
    genrl_task_accuracies = np.mean(genrl_task_accuracies, axis=0)
    
    meta_train_memory_mean = np.mean(mem_accuracy)
    meta_train_memory_std = np.std(mem_accuracy)

    meta_train_generalization_mean = np.mean(genrl_accuracy)
    meta_train_generalization_std = np.std(genrl_accuracy)



    return meta_train_memory_mean, meta_train_memory_std, meta_train_generalization_mean, meta_train_generalization_std, memory_task_accuracies, genrl_task_accuracies
        

        
       
##################################################################################################################################
eval_memory_losses = []
eval_memory_losses_std = []
def test( device, eval_data_loader, dims, n_channels, samples_per_task, num_tasks, k_steps, ax2, x_plotLength, Data_Augmenter):
    global eval_memory_losses
    global eval_memory_losses_std
    num_tasks = test_tasks

    global eval_samples_per_stream
    global eval_samples_per_task
    test_batch_size = eval_samples_per_stream
    test_train_size =  eval_samples_per_task

    

    all_mem_accs = []
    all_mem_accuracies = []
    all_genrl_accs = []
    all_genrl_accuracies = []

    tmp_mem_losses = []
    eval_HLT_memory_losses = []
    for m_idx in range(3):
   
        #--- MODEL RELOAD  (not necessary if learning a prior over the whole model) ------------------------------------
        #dan_opt_checkpoint = torch.load("noniid_hyper_learning16_ultra2_dan_opt_checkpoint_m1.pth")

        torch.manual_seed(torch.randint(0,400, (1,)).item())
        model = DAN_classifier(dims=dims, n_channels=n_channels)
        phenotype_model = DAN_classifier(dims=dims, n_channels=n_channels)
        phenotype_model.load_state_dict(torch.load("noniid_hyper_learning16_ultra2_m"+str(m_idx+1)+".pt")) 

        #load all 3 param groups if we are learning a prior over the whole model
        
        model.DAN_layers.load_state_dict(phenotype_model.DAN_layers.state_dict())
        #model.VECS_layers.load_state_dict(phenotype_model.VECS_layers.state_dict())
        
        # model.ENC_layers.load_state_dict(phenotype_model.ENC_layers.state_dict())
        #model.VECS_layers.load_state_dict(phenotype_model.VECS_layers.state_dict())
        model = model.to(device)

        vecs_optimizer = torch.optim.Adam(model.VECS_layers.parameters(),  lr=vecs_lr)
        
        dan_optimizer = torch.optim.Adam(model.DAN_layers.parameters(),  lr=dan_lr)

        del phenotype_model
        #--- END - ORIGINAL MODEL RELOAD ------------------------------------

        for batch in eval_data_loader:
            data, target = batch
            

            new_data = torch.FloatTensor([])
            new_target = torch.LongTensor([])
            for task_id in range(10):
                target_idx = target == task_id
                tmp_target = target[target_idx][0:test_batch_size]
                tmp_data = data[target_idx][0:test_batch_size]
                new_target  = torch.cat((new_target, tmp_target), axis=0)
                new_data  = torch.cat((new_data, tmp_data), axis=0)
        
            data = new_data.view(1,test_batch_size*10, 1, 32, 32)

            data = data.repeat(1,1,3,1,1)
            target = new_target.view(1,test_batch_size*10)

            #data, target = data.to(device), target.to(device)

            
            eval_memory_loss = 0


            # training_data = torch.Tensor()
            # training_targets = torch.Tensor()
            # for i in range(num_tasks):
            #     batch_start = i*test_batch_size
            #     training_data = torch.cat((training_data, data[0][batch_start: batch_start+test_train_size])) 
            #     training_targets = torch.cat((training_targets, target[0][batch_start: batch_start+test_train_size].float()))
            
            for i in range(num_tasks):
                #print("Eval Task:", i)
               
                
               
                for step in range(test_steps):
                    batch_start = i*test_batch_size
                    batch_end = batch_start + test_train_size
                    synapse_indices = np.random.randint(batch_start,batch_end, test_train_size)
                    eval_curdata = data[0][synapse_indices].view((test_train_size,3,32,32))
                    eval_cur_targets = target[0][synapse_indices].view((test_train_size))

                    
                    
                    eval_cur_targets = torch.cat((eval_cur_targets, eval_cur_targets))
                    eval_cur_targets = eval_cur_targets.long()
                    vecs_optimizer.zero_grad() 
                    dan_optimizer.zero_grad()

                    augemented_eval_data = fetch_augmented_data(eval_curdata, Data_Augmenter)
                    eval_curdata = torch.cat((eval_curdata, augemented_eval_data))
                    
                    
                    sample_idx = np.random.randint(0, eval_curdata.shape[0], 3)
                    eval_curdata = eval_curdata[sample_idx]
                    eval_cur_targets = eval_cur_targets[sample_idx]
                    eval_curdata = eval_curdata.to(device)
                    eval_cur_targets = eval_cur_targets.to(device)
                    #cur_target = one_hot_embedding(eval_cur_targets, num_hyper_train_tasks).to(device)

                    eval_output = model(eval_curdata, test_train_size*2, use_dropout=False, train_step=True)
                    
                    eval_output = torch.reshape(eval_output[:, :], (eval_output.size()[0],num_hyper_train_tasks))
                    eval_loss = F.nll_loss(eval_output, eval_cur_targets) #F.mse_loss(eval_output, eval_curdata)

                    
                    eval_loss.backward()
                    
                    vecs_optimizer.step()
                    #dan_optimizer.step()

                
                cur_mem_preds = torch.LongTensor().to(device)
                cur_mem_targets = torch.LongTensor().to(device)
                for eval_mem_idx in range(4):
                    dan_optimizer.zero_grad()
                    vecs_optimizer.zero_grad()
                    eval_hyper_indices = np.random.randint(0,  (i+1)*test_batch_size, test_train_size)
                    eval_hyper_data = Variable(data[0][eval_hyper_indices]).view((test_train_size,3,32,32))
                    #check memory
                    
                    
                    # for mem_sample in range(i+1):
                    #     batch_start = mem_sample*samples_per_task
                    # mem_data = data[0][batch_start:batch_start+samples_per_task].view((samples_per_task,3,32,32))
                    mem_targets = target[0][eval_hyper_indices].reshape((test_train_size))
                    mem_targets = mem_targets.long()
                    

                    sample_idx = np.random.randint(0, eval_hyper_data.shape[0], test_train_size)
                    eval_hyper_data = eval_hyper_data[sample_idx]
                    mem_targets = mem_targets[sample_idx]
                    eval_hyper_data = eval_hyper_data.to(device)
                    mem_targets = mem_targets.to(device)
                    #cur_target = one_hot_embedding(mem_targets, num_hyper_train_tasks).to(device)

                    mem_output = model(eval_hyper_data, eval_hyper_data.shape[0], use_dropout=False, train_step=True)
                    mem_output = torch.reshape(mem_output[:, :num_tasks], (mem_output.size()[0], num_tasks))
                    mem_preds = torch.argmax(mem_output, dim=1)
                    for z in range(9):

                        sample_idx = np.random.randint(0, eval_hyper_data.shape[0], test_train_size)
                        eval_hyper_data = eval_hyper_data[sample_idx]
                        mem_targets = mem_targets[sample_idx]
                        eval_hyper_data = eval_hyper_data.to(device)
                        mem_targets = mem_targets.to(device)

                        mem_output = model(eval_hyper_data, eval_hyper_data.shape[0], use_dropout=False, train_step=True)
                        mem_output = torch.reshape(mem_output[:, :num_tasks], (mem_output.size()[0], num_tasks))
                        mem_preds += torch.argmax(mem_output, dim=1)
                    mem_preds /= 10

                
                    cur_mem_preds = torch.cat((cur_mem_preds, mem_preds))
                    mem_targets = mem_targets.to(device)
                    cur_mem_targets = torch.cat((cur_mem_targets, mem_targets))
            

                eval_memory_loss += ((cur_mem_targets == cur_mem_preds).sum().item()/(test_train_size*4))*100. #F.nll_loss(mem_output, mem_targets) #F.mse_loss(mem_output, mem_data)
            eval_memory_loss /= num_tasks


            eval_HLT_memory_losses.append(eval_memory_loss)
            tmp_mem_losses.append(100-eval_memory_loss)
            

            meta_eval_memory_accuracy, meta_eval_memory_task_accuracies = get_meta_eval_memory_accuracy(model, m_idx, data, target, device,  eval_samples_per_task, eval_samples_per_stream)
            meta_eval_generalization_accuracy, meta_eval_generalization_task_accuracies = get_meta_eval_generalization_accuracy(model, m_idx, data, target, device,  eval_samples_per_task, eval_samples_per_stream)
            all_mem_accs.append(meta_eval_memory_accuracy)
            all_mem_accuracies.append(meta_eval_memory_task_accuracies)

            all_genrl_accs.append(meta_eval_generalization_accuracy)
            all_genrl_accuracies.append(meta_eval_generalization_task_accuracies)

            break
        model.cpu()
    meta_eval_memory_task_accuracies = np.array(all_mem_accuracies).reshape(len(all_mem_accuracies), num_tasks)
    meta_eval_memory_task_accuracies = np.mean(meta_eval_memory_task_accuracies, axis=0)

    meta_eval_generalization_task_accuracies = np.array(all_genrl_accuracies).reshape(len(all_genrl_accuracies), num_tasks)
    meta_eval_generalization_task_accuracies = np.mean(meta_eval_generalization_task_accuracies, axis=0)

    meta_eval_memory_accuracy = np.mean(all_mem_accs)
    meta_eval_memory_std = np.std(all_mem_accs)

    meta_eval_generalization_accuracy = np.mean(all_genrl_accs)
    meta_eval_generalization_std = np.std(all_genrl_accs)

    eval_memory_losses.append(np.mean(tmp_mem_losses))
    eval_memory_losses_std.append(np.std(tmp_mem_losses))
    
    line2, = ax2.plot(x_plotLength, eval_memory_losses, c="blue", label="Meta-Eval")
    
    ax2.set_ylabel('Meta-Eval Memory Loss', color="blue")  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor="blue")
    ax2.set_ylim(0)
    

    return line2, eval_memory_losses, eval_memory_losses_std, meta_eval_memory_accuracy, meta_eval_memory_std, meta_eval_generalization_accuracy, meta_eval_generalization_std, meta_eval_memory_task_accuracies, meta_eval_generalization_task_accuracies


            
            
##################################################################################################################################

   


def main():

    global loss_scaler
    global my_batch_size
    samples_per_task = my_batch_size
    global num_tasks 
    global num_img_tasks
    global test_tasks 
    global k_steps 


    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    torch.cuda.set_device(0)
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    

    Data_Augmenter = Single_Channel_Transform()

    dataset = omniglot("data", ways=num_hyper_train_CL_tasks, shots=samples_per_stream, test_shots=(20-samples_per_stream), meta_train=True, download=True, shuffle=True, transform=transform)

    omniglot_test_dataset = omniglot("data", ways=num_tasks, shots=samples_per_stream, test_shots=(20-samples_per_stream), meta_test=True, download=True, shuffle=True, transform=transform)
    
    # dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=5, num_test_per_class=15)
    data_loader = BatchMetaDataLoader(dataset, batch_size=k_streams, num_workers=4)

    omniglot_test_data_loader = BatchMetaDataLoader(omniglot_test_dataset, batch_size=k_streams, num_workers=4)

     #------  MNIST -----------------------------------------------------------------
    train_kwargs = {'batch_size': 2000}
    eval_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    eval_data_loader = torch.utils.data.DataLoader(eval_dataset,**train_kwargs)
    #------  MNIST -----------------------------------------------------------------

    """
        I need m models, 2m optimizers, store in dict
        I need m streams of data, maybe also store in the dict or keep separate in list
        I need to keep the VECS unique, but share DANs among all models AT EVERY STEP!
        I therefore need to make a pass through the data using the correct data stream for each 
            model and each set of params (VECS and DANs)
            - I need to collect the corresponding gradients for VECS and DANs
            - I need to average the gradients for DANs and apply that update to each of the models.
            - Need to apply the unique gradients for VECS to each of the models.
    """
    ############  MODEL ############################################################################
    #This works pretty good for now
    n_channels = 100
    target_size = num_hyper_train_tasks
    dims=[32*32, 40, 20, 20, num_hyper_train_tasks]
  
    models = {}
   
    for m_idx in range(num_models):
        models["m"+str(m_idx+1)] = {} #model
        torch.manual_seed(torch.randint(0,400, (1,)).item())
        models["m"+str(m_idx+1)]["model"] = DAN_classifier(dims=dims, n_channels=n_channels)

        size = 0
        for n,p in models["m"+str(m_idx+1)]["model"].VECS_layers.named_parameters():
            #print(torch.flatten(p).shape[0])
            size += torch.flatten(p).shape[0]
        print("Size: ", f"{size:,}")
        # sys.exit()

        models["m"+str(m_idx+1)]["vecs_opt"] = torch.optim.Adam(models["m"+str(m_idx+1)]["model"].VECS_layers.parameters(),  lr=vecs_lr) 
        models["m"+str(m_idx+1)]["dan_opt"] = torch.optim.Adam(models["m"+str(m_idx+1)]["model"].DAN_layers.parameters(),  lr=dan_lr) 

    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, 40):
        train(args, models, device, data_loader, epoch, dims, n_channels, eval_data_loader, omniglot_test_data_loader, Data_Augmenter)
        #test(model, device, data_loader)
        #scheduler.step()

        #if args.save_model:
        


if __name__ == '__main__':
    main()
