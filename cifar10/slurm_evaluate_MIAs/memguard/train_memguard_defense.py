import argparse
import os
parser = argparse.ArgumentParser() 
parser.add_argument('--config', type=str, default='./configs/memguard.yml')  
parser.add_argument('--world-size', type=int, required=True)  
parser.add_argument('--diff', type=int, default=0)  
parser.add_argument('--mode', type=int, default=0)  
args = parser.parse_args()
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.distributions import Categorical
import datetime
import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import math
import sys
from sklearn.metrics import roc_auc_score, roc_curve, auc
sys.path.append('../util') 
sys.path.append('../') 
# sys.path.insert(0,'./util/')
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import * 
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from util.densenet import densenet
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc 
from score_based_MIA_util import black_box_benchmarks
from diff_defense.diff_warpper import * 
from tqdm import tqdm as tq
from scipy.special import softmax
from dist_attack import get_phi, aggregate_predictions, KL
os.chdir('../')

def parse_config(config_path=None):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        new_config = dict2namespace(config)
    return new_config
config = parse_config(args.config)
num_classes = config.trainer.num_class

num_sample=config.attack.num_sample
world_size = args.world_size
num_per_rank = math.ceil(num_sample/world_size)
BATCH_SIZE=256
if not args.diff == 0:
    data_path = f'./slurm_evaluate_MIAs/data/{config.attack.target_model}/num_sample{config.attack.num_sample}/{config.attack.save_tag}/diff{args.diff}/iter{config.purification.max_iter}_path{config.purification.path_number}_step{config.purification.purify_step}'
else:
    data_path = f'./slurm_evaluate_MIAs/data/{config.attack.target_model}/num_sample{config.attack.num_sample}/{config.attack.save_tag}/diff{args.diff}'

import time
start = time.time()


def get_outputs_labels():
    shadow_train_performance_outputs=[]
    shadow_train_performance_labels=[]
    shadow_test_performance_outputs=[]
    shadow_test_performance_labels=[]
    target_train_performance_outputs=[]
    target_train_performance_labels=[]
    target_test_performance_outputs=[]
    target_test_performance_labels=[]

    test_target_train_performance_outputs=[]
    test_target_train_performance_labels=[]
    test_target_test_performance_outputs=[]
    test_target_test_performance_labels=[]
    
    for rank in range(world_size):
        path = os.path.join(data_path, f'diff_{world_size}_{rank}.npz')
        data = np.load(path)
        shadow_train_performance_outputs.append(data['shadow_train_performance_logits'])
        shadow_train_performance_labels.append(data['shadow_train_performance_glabels'])
        shadow_test_performance_outputs.append(data['shadow_test_performance_logits'])
        shadow_test_performance_labels.append(data['shadow_test_performance_glabels'])
        target_train_performance_outputs.append(data['target_train_performance_logits'])
        target_train_performance_labels.append(data['target_train_performance_glabels'])
        target_test_performance_outputs.append(data['target_test_performance_logits'])
        target_test_performance_labels.append(data['target_test_performance_glabels'])

        test_target_train_performance_outputs.append(data['test_target_train_performance_logits'])
        test_target_train_performance_labels.append(data['test_target_train_performance_glabels'])
        test_target_test_performance_outputs.append(data['test_target_test_performance_logits'])
        test_target_test_performance_labels.append(data['test_target_test_performance_glabels'])

    shadow_train_performance_outputs=softmax(np.concatenate(shadow_train_performance_outputs),1)
    shadow_train_performance_labels=np.concatenate(shadow_train_performance_labels)
    shadow_test_performance_outputs=softmax(np.concatenate(shadow_test_performance_outputs),1)
    shadow_test_performance_labels=np.concatenate(shadow_test_performance_labels)
    target_train_performance_outputs=softmax(np.concatenate(target_train_performance_outputs),1)
    target_train_performance_labels=np.concatenate(target_train_performance_labels)
    target_test_performance_outputs=softmax(np.concatenate(target_test_performance_outputs),1)
    target_test_performance_labels=np.concatenate(target_test_performance_labels)

    test_target_train_performance_outputs=softmax(np.concatenate(test_target_train_performance_outputs),1)
    test_target_train_performance_labels=np.concatenate(test_target_train_performance_labels)
    test_target_test_performance_outputs=softmax(np.concatenate(test_target_test_performance_outputs),1)
    test_target_test_performance_labels=np.concatenate(test_target_test_performance_labels)  


    return (test_target_train_performance_outputs,test_target_train_performance_labels),\
        (test_target_test_performance_outputs,test_target_test_performance_labels),\
        (shadow_train_performance_outputs,shadow_train_performance_labels),\
        (shadow_test_performance_outputs,shadow_test_performance_labels),\
        (target_train_performance_outputs,target_train_performance_labels),\
        (target_test_performance_outputs,target_test_performance_labels)

def accuracy(output, target):
    def accuracy1(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    top1 = AverageMeter()
    top5 = AverageMeter()

    prec1, prec5 = accuracy1(output, target, topk=(1, 5))
    top1.update(prec1.item(), len(output))
    top5.update(prec5.item(), len(output))

    return (top1.avg, top5.avg)


test_target_train_performance,test_target_test_performance,\
    shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance =get_outputs_labels()

#YF
if args.diff!=0:
    mode = args.mode
    SCOPE = config.attack.scope
    # draw the plot of dist for members and non-members
    num=1000
    def get_same_prediction(outputs,labels):
        all_outputs=[]
        for i in range(len(outputs)):
            all_outputs.extend(outputs[i][:-1][np.argmax(outputs[i][:-1],1)==labels[i]])
        return np.array(all_outputs)

    mem_out = get_same_prediction(shadow_train_performance[0][:num],shadow_train_performance[1][:num])
    nonmem_out = get_same_prediction(shadow_test_performance[0][:num],shadow_test_performance[1][:num])

    # print(mem_out.shape,nonmem_out.shape)
    import matplotlib.pyplot as plt

    mem_logits, nonmem_logits = get_phi(mem_out), get_phi(nonmem_out)
    plt.figure(figsize=(8,5), dpi= 100)

    bins = np.histogram_bin_edges(nonmem_logits, bins=100)
    # logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

    # print("logit median:",np.median(mem_logits),np.median(nonmem_logits))

    n1,bins1,_ = plt.hist(mem_logits, bins=bins, label="Train Samples", alpha=.7)
    n2,bins2,_ = plt.hist(nonmem_logits, bins=bins, label="Test Samples", alpha=.7)
    kl = KL(n1,n2)
    scope1=[np.min(mem_logits),np.max(nonmem_logits)]
    # print('scope',scope1)
    # print('original KL', kl)

    # print("logit mode:",bins1[np.argmax(n1)], bins2[np.argmax(n2)])
    plt.xlabel("Logit of the Confidences",fontsize=15)#横坐标名字
    plt.ylabel("#Samples",fontsize=15)
    # plt.title(f'dist',fontsize=20)
    plt.legend(loc = "best",fontsize=15)
    plt.savefig(f'{data_path}/logit_dist.png',dpi=300,bbox_inches='tight')

    if mode == 2:
        SCOPE=[np.min(mem_logits),np.mean(mem_logits)]
    elif mode == 3:
        SCOPE=[np.min(nonmem_logits),np.max(mem_logits)]

    if SCOPE == None:
        all_scopes=[]
        all_kl=[]  
        # print('mean',np.mean(mem_logits),np.mean(nonmem_logits))
        # print('median',np.median(mem_logits),np.median(nonmem_logits))
        # scope=[bins2[np.argmax(n2)],bins1[np.argmax(n1)]]
        scope=[np.min(mem_logits),np.max(nonmem_logits)]
        # print('scope',scope)
        step = (scope[1]-scope[0])/20
        s_max = scope[1]
        s_min = scope[0]
        for i in np.arange(s_min,s_max+step,step):
            for j in np.arange(s_min,s_max+step,step):
                if j<=i: continue
                scope = [i,j]            
                all_scopes.append([i,j])
            
                temp_shadow_train_performance = aggregate_predictions(shadow_train_performance[0],shadow_train_performance[1],scope, config), shadow_train_performance[1], shadow_train_performance[2]
                temp_shadow_test_performance = aggregate_predictions(shadow_test_performance[0],shadow_test_performance[1],scope, config), shadow_test_performance[1], shadow_test_performance[2]

                #draw plot after defense
                mem_logits, nonmem_logits = get_phi(temp_shadow_train_performance[0]), get_phi(temp_shadow_test_performance[0])

                bins = np.histogram_bin_edges(nonmem_logits, bins=100)
                # logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
                # print("logit median:",np.median(mem_logits),np.median(nonmem_logits))
                n1,bins1,_ = plt.hist(mem_logits, bins=bins, label="Train Samples", alpha=.7)
                n2,bins2,_ = plt.hist(nonmem_logits, bins=bins, label="Test Samples", alpha=.7)
                kl = KL(n1,n2)
                # print('KL', kl)
                all_kl.append(kl)

        SCOPE = all_scopes[np.argmin(all_kl)]
    # print("SCOPE:",SCOPE)
   
    test_target_train_performance = aggregate_predictions(test_target_train_performance[0],test_target_train_performance[1],SCOPE, config),test_target_train_performance[1], test_target_train_performance[2]
    test_target_test_performance = aggregate_predictions(test_target_test_performance[0],test_target_test_performance[1],SCOPE, config), test_target_test_performance[1], test_target_test_performance[2]
    shadow_train_performance = aggregate_predictions(shadow_train_performance[0],shadow_train_performance[1],SCOPE, config), shadow_train_performance[1], shadow_train_performance[2]
    shadow_test_performance = aggregate_predictions(shadow_test_performance[0],shadow_test_performance[1],SCOPE, config), shadow_test_performance[1], shadow_test_performance[2]
    target_train_performance = aggregate_predictions(target_train_performance[0],target_train_performance[1],SCOPE, config), target_train_performance[1], target_train_performance[2]
    target_test_performance = aggregate_predictions(target_test_performance[0],target_test_performance[1],SCOPE, config), target_test_performance[1], target_test_performance[2]


shadow_train_performance = (np.concatenate([shadow_train_performance[0],target_train_performance[0]]),\
                            np.concatenate([shadow_train_performance[1],target_train_performance[1]]))

shadow_test_performance = (np.concatenate([shadow_test_performance[0],target_test_performance[0]]),\
                            np.concatenate([shadow_test_performance[1],target_test_performance[1]]))

# shadow_test_performance = test_target_test_performance

print(shadow_train_performance[0].shape,shadow_test_performance[0].shape)

# if(config.attack.getModelAcy):
#     import copy
#     check_test_acc = accuracy(torch.from_numpy(test_target_test_performance[0]),torch.from_numpy(test_target_test_performance[1]))[0]
#     check_val_acc = accuracy(torch.from_numpy(test_target_val_performance[0]),torch.from_numpy(test_target_val_performance[1]))[0]
#     check_train_acc = accuracy(torch.from_numpy(test_target_train_performance[0]),torch.from_numpy(test_target_train_performance[1]))[0]
#     print('{} | train acc {:.4f} | val acc {:.4f} | test acc {:.4f}'.format(config.attack.path, check_train_acc,check_val_acc,check_test_acc), flush=True)

import tensorflow.keras as keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, concatenate
import numpy as np
import tensorflow as tf
np.random.seed(10000)
print(tf.test.is_gpu_available())

defense_epochs=200
batch_size=128
defense_num_classes=1

def model_defense(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(inputs_b)
    x_b=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    x_b=Dense(64,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model

f_train = np.concatenate((shadow_train_performance[0],shadow_test_performance[0]))
y_train = np.concatenate((shadow_train_performance[1],shadow_test_performance[1]))
y_train=tf.keras.utils.to_categorical(y_train,num_classes)

l_train=np.zeros([f_train.shape[0]],dtype=np.int)
l_train[0:shadow_train_performance[0].shape[0]]=1

print(f_train.shape,y_train.shape,l_train.shape)

f_train=np.sort(f_train,axis=1)
input_shape=y_train.shape[1:]

model=model_defense(input_shape=input_shape,labels_dim=defense_num_classes)
model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.SGD(lr=0.001),metrics=['accuracy'])
model.summary()

b_train=f_train[:,:]
label_train=l_train[:]

index_array=np.arange(b_train.shape[0])
batch_num=np.int(np.ceil(b_train.shape[0]/batch_size))


for i in np.arange(defense_epochs):
    np.random.shuffle(index_array)
    for j in np.arange(batch_num):
        b_batch=b_train[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,b_train.shape[0])],:]
        y_batch=label_train[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,label_train.shape[0])]]
        model.train_on_batch(b_batch,y_batch)   

    if (i+1)%50==0:
        print("Epochs: {}".format(i))
        scores_train = model.evaluate(b_train, label_train, verbose=0)
        print('Train loss:', scores_train[0])
        print('Train accuracy:', scores_train[1])    
    
defense_model_path = f'./slurm_evaluate_MIAs/memguard/{config.attack.target_model}'
# defense_model_path = os.path.join(defense_model_path, f'memguard_diff{args.diff}_{config.attack.save_tag}')
if not os.path.exists(defense_model_path ):
        os.makedirs(defense_model_path)

if args.diff==0:
    model.save(os.path.join(defense_model_path, f'memguard_diff{args.diff}_{config.attack.save_tag}'))
else:
    model.save(os.path.join(defense_model_path, f'memguard_diff{args.diff}_{config.attack.save_tag}_mode{args.mode}'))
print(model.predict(b_train[:10]))
print(sum(label_train))
# weights=model.get_weights()
# np.savez(os.path.join(defense_model_path, f'memguard_diff{args.diff}_{config.attack.save_tag}.npz'),x=weights)

end = time.time()
print('running time:', end-start)
  