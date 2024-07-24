import argparse
import os
parser = argparse.ArgumentParser() 
parser.add_argument('--config', type=str, default='./configs/default.yml')  
parser.add_argument('--world-size', type=int, required=True)  
parser.add_argument('--diff', type=int, default=0)  
parser.add_argument('--mode', type=int, default=0)  
parser.add_argument('--N', type=int, default=0) 
parser.add_argument('--T', type=int, default=0) 
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
from model_factory import *
from scipy.special import softmax
import scipy
os.chdir('../')

def parse_config(config_path=None):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        new_config = dict2namespace(config)
    return new_config
config = parse_config(args.config)

if args.mode==0:
    mode = config.attack.mode
else:
    mode = args.mode

if not args.N == 0:
    config.purification.path_number = args.N
if not args.T == 0:
    config.purification.purify_step = args.T


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_sample=config.attack.num_sample
world_size = args.world_size
num_per_rank = math.ceil(num_sample/world_size)
num_classes = config.trainer.num_class

BATCH_SIZE=128
if not args.diff == 0:
    data_path = f'./slurm_evaluate_MIAs/data/{config.attack.target_model}/num_sample{config.attack.num_sample}/{config.attack.save_tag}/diff{args.diff}/iter{config.purification.max_iter}_path{config.purification.path_number}_step{config.purification.purify_step}'
else:
    data_path = f'./slurm_evaluate_MIAs/data/{config.attack.target_model}/num_sample{config.attack.num_sample}/{config.attack.save_tag}/diff{args.diff}'

import time
start = time.time()

criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()

def get_tpr(y_true, y_score, fpr_threshold=0.001, attack='None'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    accuracy = np.max(1-(fpr+(1-tpr))/2)
    auc_score = auc(fpr, tpr)
    highest_tpr = tpr[np.where(fpr<fpr_threshold)[0][-1]]

    ## TPR and FPR are in ascending order
    ## TNR and FNR are in descending order
    fnr = 1 - tpr 
    tnr = 1 - fpr
    highest_tnr = tnr[np.where(fnr<fpr_threshold)[0][0]] 
    print( '\t\t===> %s: TPR %.2f%% @%.2f%%FPR | TNR %.2f%% @%.2f%%FNR | AUC %.4f| ACC %.4f'%( attack, highest_tpr*100, fpr_threshold*100, highest_tnr*100, fpr_threshold*100, auc_score, accuracy  ) )
    return auc_score, accuracy
    # best_pred=[]
    # best_acc=0
    # for thre in thresholds:
    #     pred = [1 if y>thre else 0 for y in y_score ]
    #     if np.mean(pred==y_true)>best_acc:
    #         best_acc = np.mean(pred==y_true)
    #         best_pred = pred
    # print(classification_report(y_true,best_pred))

def get_logits_labels():
    shadow_train_performance_logits=[]
    shadow_train_performance_plabels=[]
    shadow_train_performance_glabels=[]

    shadow_test_performance_logits=[]
    shadow_test_performance_plabels=[]
    shadow_test_performance_glabels=[]

    target_train_performance_logits=[]
    target_train_performance_plabels=[]
    target_train_performance_glabels=[]

    target_test_performance_logits=[]
    target_test_performance_plabels=[]
    target_test_performance_glabels=[]

    test_target_train_performance_logits=[]
    test_target_train_performance_plabels=[]
    test_target_train_performance_glabels=[]

    test_target_test_performance_logits=[]
    test_target_test_performance_plabels=[]
    test_target_test_performance_glabels=[]
    
    for rank in range(world_size):
        path = os.path.join(data_path, f'diff_{world_size}_{rank}.npz')
        data = np.load(path)
        shadow_train_performance_logits.append(data['shadow_train_performance_logits'])
        shadow_train_performance_plabels.append(data['shadow_train_performance_plabels'])
        shadow_train_performance_glabels.append(data['shadow_train_performance_glabels'])
        shadow_test_performance_logits.append(data['shadow_test_performance_logits'])
        shadow_test_performance_plabels.append(data['shadow_test_performance_plabels'])
        shadow_test_performance_glabels.append(data['shadow_test_performance_glabels'])
        target_train_performance_logits.append(data['target_train_performance_logits'])
        target_train_performance_plabels.append(data['target_train_performance_plabels'])
        target_train_performance_glabels.append(data['target_train_performance_glabels'])
        target_test_performance_logits.append(data['target_test_performance_logits'])
        target_test_performance_plabels.append(data['target_test_performance_plabels'])
        target_test_performance_glabels.append(data['target_test_performance_glabels'])

        test_target_train_performance_logits.append(data['test_target_train_performance_logits'])
        test_target_train_performance_plabels.append(data['test_target_train_performance_plabels'])
        test_target_train_performance_glabels.append(data['test_target_train_performance_glabels'])
        test_target_test_performance_logits.append(data['test_target_test_performance_logits'])
        test_target_test_performance_plabels.append(data['test_target_test_performance_plabels'])
        test_target_test_performance_glabels.append(data['test_target_test_performance_glabels'])

    shadow_train_performance_logits=np.concatenate(shadow_train_performance_logits)
    shadow_train_performance_plabels=np.concatenate(shadow_train_performance_plabels)
    shadow_train_performance_glabels=np.concatenate(shadow_train_performance_glabels)

    shadow_test_performance_logits=np.concatenate(shadow_test_performance_logits)
    shadow_test_performance_plabels=np.concatenate(shadow_test_performance_plabels)
    shadow_test_performance_glabels=np.concatenate(shadow_test_performance_glabels)

    target_train_performance_logits=np.concatenate(target_train_performance_logits)
    target_train_performance_plabels=np.concatenate(target_train_performance_plabels)
    target_train_performance_glabels=np.concatenate(target_train_performance_glabels)

    target_test_performance_logits=np.concatenate(target_test_performance_logits)
    target_test_performance_plabels=np.concatenate(target_test_performance_plabels)
    target_test_performance_glabels=np.concatenate(target_test_performance_glabels)

    test_target_train_performance_logits=np.concatenate(test_target_train_performance_logits)
    test_target_train_performance_plabels=np.concatenate(test_target_train_performance_plabels)
    test_target_train_performance_glabels=np.concatenate(test_target_train_performance_glabels)

    test_target_test_performance_logits=np.concatenate(test_target_test_performance_logits)
    test_target_test_performance_plabels=np.concatenate(test_target_test_performance_plabels) 
    test_target_test_performance_glabels=np.concatenate(test_target_test_performance_glabels) 
    
    return (test_target_train_performance_logits,test_target_train_performance_plabels,test_target_train_performance_glabels),\
        (test_target_test_performance_logits,test_target_test_performance_plabels,test_target_test_performance_glabels),\
        (shadow_train_performance_logits,shadow_train_performance_plabels,shadow_train_performance_glabels),\
        (shadow_test_performance_logits,shadow_test_performance_plabels,shadow_test_performance_glabels),\
        (target_train_performance_logits,target_train_performance_plabels,target_train_performance_glabels),\
        (target_test_performance_logits,target_test_performance_plabels,target_test_performance_glabels)


def get_outputs_labels_memguard(islogits=False):
    shadow_train_performance_outputs=[]
    shadow_test_performance_outputs=[]
    target_train_performance_outputs=[]
    target_test_performance_outputs=[]

    for rank in range(world_size):
        if islogits==False:
            shadow_train_performance_outputs.append(np.load(os.path.join(data_path, 'memguard_defense_results', f'memguard_known_member_{world_size}_{rank}.npy')))
            shadow_test_performance_outputs.append(np.load(os.path.join(data_path, f'memguard_defense_results', f'memguard_known_nonmember_{world_size}_{rank}.npy')))
            target_train_performance_outputs.append(np.load(os.path.join(data_path, f'memguard_defense_results', f'memguard_test_member_{world_size}_{rank}.npy')))
            target_test_performance_outputs.append(np.load(os.path.join(data_path, f'memguard_defense_results', f'memguard_test_non_member_{world_size}_{rank}.npy')))
        else:
            shadow_train_performance_outputs.append(np.load(os.path.join(data_path, 'memguard_defense_results', f'memguard_known_member_logit_{world_size}_{rank}.npy')))
            shadow_test_performance_outputs.append(np.load(os.path.join(data_path, f'memguard_defense_results', f'memguard_known_nonmember_logit_{world_size}_{rank}.npy')))
            target_train_performance_outputs.append(np.load(os.path.join(data_path, f'memguard_defense_results', f'memguard_test_member_logit_{world_size}_{rank}.npy')))
            target_test_performance_outputs.append(np.load(os.path.join(data_path, f'memguard_defense_results', f'memguard_test_non_member_logit_{world_size}_{rank}.npy')))

    shadow_train_performance_outputs=np.concatenate(shadow_train_performance_outputs)
    shadow_test_performance_outputs=np.concatenate(shadow_test_performance_outputs)
    target_train_performance_outputs=np.concatenate(target_train_performance_outputs)
    target_test_performance_outputs=np.concatenate(target_test_performance_outputs)

    return shadow_train_performance_outputs,shadow_test_performance_outputs,target_train_performance_outputs,target_test_performance_outputs

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

#from https://github.com/sirius8050/Expected-Calibration-Error/blob/master/ECE.py
def ece_score(py, y_test, n_bins=10):
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)

def memguard_get_pred_loss(preds, y):
    criterion = nn.CrossEntropyLoss(reduction='none') 
    one_hot = np.zeros( (len(y), num_classes) )
    for i in range( len(y) ):
        one_hot[i] = tf.keras.utils.to_categorical( y[i], num_classes=num_classes) 
    loss = np.asarray([-math.log(y_pred) if y_pred > 0 else y_pred+1e-50 for y_pred in preds[one_hot.astype(bool) ] ])
    return loss

test_target_train_performance,test_target_test_performance,\
    shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance =get_logits_labels()

def KL(p,q):
    p = np.clip(p,a_min=10e-20,a_max=np.max(p))
    q = np.clip(q,a_min=10e-20,a_max=np.max(q))
    # return(scipy.stats.entropy(p,q,base=2)) 
    return scipy.spatial.distance.jensenshannon(p,q)

def get_phi(opredictions):

    opredictions = np.expand_dims(opredictions,axis=1)
    labels = np.argmax(opredictions,axis=2).flatten()
    ## Be exceptionally careful.
    ## Numerically stable everything, as described in the paper.

    predictions = opredictions - np.max(opredictions, axis=2, keepdims=True)

    predictions = np.array(np.exp(predictions), dtype=np.float64)
    predictions = predictions/np.sum(predictions,axis=2,keepdims=True)
    COUNT = predictions.shape[0]
    y_true = predictions[np.arange(COUNT),:,labels[:COUNT]]

    predictions[np.arange(COUNT),:,labels[:COUNT]] = 0
    y_wrong = np.sum(predictions, axis=2)
    logit = (np.log(y_true.mean((1))+1e-45) - np.log(y_wrong.mean((1))+1e-45))
    return logit

def get_confidence(opredictions):
    return np.max(softmax(opredictions,1),1)

def get_entropy(opredictions,num_classes=10):
    opredictions = softmax(opredictions,1)
    return np.sum(-opredictions* np.log(opredictions),axis=1)/num_classes

def aggregate_predictions(predictions,p_labels, scope, config):

    # scope = SCOPE
    N,P,C=predictions.shape
    predictions_flat = predictions.reshape(N*P,C)
    logits = get_phi(predictions_flat).reshape(N,P)

    fail=0
    sample_pred=[]
    for i in range(len(predictions)):

        sample_all_path=predictions[i]
        label = p_labels[i]
        logit = logits[i]

        logit = logit[np.argmax(sample_all_path,1)==label]
        sample_all_path = sample_all_path[np.argmax(sample_all_path,1)==label]

        for j in range(len(sample_all_path)):
            if logit[j] <=scope[1] and logit[j] >=scope[0]:
                sample_pred.append(sample_all_path[j])
                break
            if j==len(sample_all_path)-1:
                selected_sample=sample_all_path[np.argmin(np.abs(np.mean(scope)-logit))]
                sample_pred.append(selected_sample)
                fail+=1

    # print('fail:',fail,len(predictions))
    sample_pred = np.array(sample_pred)
    # sample_pred = softmax(sample_pred,1)
    return sample_pred 

if args.diff!=0:
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

if args.diff!=0:
    memguard_logit_path=os.path.join(data_path, f'logits_for_memguard_{mode}.npz')  
else:
    memguard_logit_path=os.path.join(data_path, f'logits_for_memguard_diff0.npz')
np.savez( memguard_logit_path,\
                shadow_train_performance_logits=shadow_train_performance[0],\
                    shadow_test_performance_logits=shadow_test_performance[0],target_train_performance_logits=target_train_performance[0],\
                    target_test_performance_logits=target_test_performance[0])
            
#draw plot after defense
plt.figure(figsize=(8,5), dpi= 100)

# mem_logits, nonmem_logits = get_confidence(shadow_train_performance[0]), get_confidence(shadow_test_performance[0])
mem_logits, nonmem_logits = get_phi(shadow_train_performance[0]), get_phi(shadow_test_performance[0])
# mem_logits, nonmem_logits = get_entropy(shadow_train_performance[0]), get_entropy(shadow_test_performance[0])


bins = np.histogram_bin_edges(nonmem_logits, bins=100)
# logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
# print("logit median:",np.median(mem_logits),np.median(nonmem_logits))
n1,bins1,_ = plt.hist(mem_logits, bins=bins, label="Training Samples", alpha=.7)
n2,bins2,_ = plt.hist(nonmem_logits, bins=bins, label="Test Samples", alpha=.7)
kl = KL(n1,n2)
# print('best KL', kl)
# if args.diff!=0:
#     print('best scope',SCOPE)
# plt.xlabel("The Highest Confidence",fontsize=15)#横坐标名字
# plt.xlabel("Prediction Entropy",fontsize=15)#横坐标名字
plt.xlabel("Logit of the Confidences",fontsize=15)#横坐标名字
plt.ylabel("#Samples",fontsize=15)
# plt.title(f'dist',fontsize=20)
plt.legend(loc = "best",fontsize=15)
plt.savefig(f'{data_path}/logit_dist_{args.diff}_{mode}.png',dpi=300,bbox_inches='tight')

test_target_train_performance = softmax(test_target_train_performance[0],1),test_target_train_performance[2]
test_target_test_performance = softmax(test_target_test_performance[0],1),test_target_test_performance[2]
shadow_train_performance = softmax(shadow_train_performance[0],1),shadow_train_performance[2]
shadow_test_performance = softmax(shadow_test_performance[0],1),shadow_test_performance[2]
target_train_performance = softmax(target_train_performance[0],1),target_train_performance[2]
target_test_performance = softmax(target_test_performance[0],1),target_test_performance[2]

def get_outputs_labels_memguard(islogits=False):
    shadow_train_performance_outputs=[]
    shadow_test_performance_outputs=[]
    target_train_performance_outputs=[]
    target_test_performance_outputs=[]

    if args.diff!=0:
        scope = SCOPE
        file_path=os.path.join(data_path, 'memguard_defense_results',f'memguard_{mode}')
    else:
        file_path=os.path.join(data_path, 'memguard_defense_results',f'memguard_diff0')
    
    for rank in range(world_size):
        if islogits==False:
            shadow_train_performance_outputs.append(np.load(os.path.join(file_path, f'memguard_known_member_{world_size}_{rank}.npy')))
            shadow_test_performance_outputs.append(np.load(os.path.join(file_path, f'memguard_known_nonmember_{world_size}_{rank}.npy')))
            target_train_performance_outputs.append(np.load(os.path.join(file_path, f'memguard_test_member_{world_size}_{rank}.npy')))
            target_test_performance_outputs.append(np.load(os.path.join(file_path, f'memguard_test_non_member_{world_size}_{rank}.npy')))
        else:
            shadow_train_performance_outputs.append(np.load(os.path.join(file_path, f'memguard_known_member_logit_{world_size}_{rank}.npy')))
            shadow_test_performance_outputs.append(np.load(os.path.join(file_path, f'memguard_known_nonmember_logit_{world_size}_{rank}.npy')))
            target_train_performance_outputs.append(np.load(os.path.join(file_path, f'memguard_test_member_logit_{world_size}_{rank}.npy')))
            target_test_performance_outputs.append(np.load(os.path.join(file_path, f'memguard_test_non_member_logit_{world_size}_{rank}.npy')))


    shadow_train_performance_outputs=np.concatenate(shadow_train_performance_outputs)
    shadow_test_performance_outputs=np.concatenate(shadow_test_performance_outputs)
    target_train_performance_outputs=np.concatenate(target_train_performance_outputs)
    target_test_performance_outputs=np.concatenate(target_test_performance_outputs)

    return shadow_train_performance_outputs,shadow_test_performance_outputs,target_train_performance_outputs,target_test_performance_outputs


if(config.attack.isMemGuard):
    file_path='./'
    print("\t====> loading MemGuard's modified output")

    shadow_train_performance_outputs,shadow_test_performance_outputs,target_train_performance_outputs,target_test_performance_outputs =get_outputs_labels_memguard(islogits=False)

    shadow_train_performance = ( shadow_train_performance_outputs, shadow_train_performance[1] )
    shadow_test_performance = ( shadow_test_performance_outputs, shadow_test_performance[1] )

    target_train_performance = ( target_train_performance_outputs, target_train_performance[1] )
    target_test_performance = ( target_test_performance_outputs, target_test_performance[1] )


if(config.attack.getModelAcy):
    import copy
    check_test_acc = accuracy(torch.from_numpy(test_target_test_performance[0]),torch.from_numpy(test_target_test_performance[1]))[0]
    check_train_acc = accuracy(torch.from_numpy(test_target_train_performance[0]),torch.from_numpy(test_target_train_performance[1]))[0]
    print('{} | train acc {:.4f} | test acc {:.4f}'.format(config.attack.path, check_train_acc,check_test_acc), flush=True)

    all_x = np.concatenate([target_train_performance[0], target_test_performance[0]])
    all_y = np.concatenate([target_train_performance[1], target_test_performance[1]])
    ece_all = ece_score(all_x, all_y)
    ece = ece_score(target_test_performance[0], target_test_performance[1])
    # print(f'|ece score {ece:.4f}|')


attack_aucs=[]
attack_accs=[]
if(config.attack.attack == 'nn' or config.attack.attack == 'all' ):

    class InferenceAttack_BB(nn.Module):
        def __init__(self,num_classes):
            self.num_classes=num_classes
            super(InferenceAttack_BB, self).__init__()
            
            self.features=nn.Sequential(
                nn.Linear(num_classes,1024),
                nn.ReLU(),
                nn.Linear(1024,512),
                nn.ReLU(),
                nn.Linear(512,64),
                nn.ReLU(),
                )

            self.labels=nn.Sequential(
               nn.Linear(num_classes,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU(),
                )

            self.loss=nn.Sequential(
               nn.Linear(1,num_classes),
                nn.ReLU(),
                nn.Linear(num_classes,64),
                nn.ReLU(),
                )
            
            self.combine=nn.Sequential(
                nn.Linear(64*3,256), 
                nn.ReLU(),
                nn.Linear(256,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Linear(64,1),
                )

            for key in self.state_dict():
                # print (key)
                if key.split('.')[-1] == 'weight':    
                    nn.init.normal_(self.state_dict()[key], std=0.01)
                    
                elif key.split('.')[-1] == 'bias':
                    self.state_dict()[key][...] = 0
            self.output= nn.Sigmoid()
        
        def forward(self,x1,one_hot_labels,loss):

            out_x1 = self.features(x1)
            
            out_l = self.labels(one_hot_labels)
            
            out_loss= self.loss(loss)

            is_member =self.combine( torch.cat((out_x1,out_l,out_loss),1))
            
            return self.output(is_member)

    def cross_entropy_loss(input, target):
        input = torch.clip(input, min=1.e-31)
        input = torch.log(input)
        target = F.one_hot(target, num_classes=num_classes)
        loss = -torch.sum(input * target,dim=1)
        return loss

    def attack_bb(inference_model, classifier_criterion, classifier_criterion_noreduct, criterion_attck, 
                  optimizer, epoch, num_batchs=1000, is_train=False, batch_size=64, non_Mem_Generator=None, eval_set_identifier_4_memguard='train'):
        global best_acc

        losses = AverageMeter()
        top1 = AverageMeter()
        mtop1_a = AverageMeter()
        mtop5_a = AverageMeter()

        acys = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        F1_Scores = AverageMeter()

        inference_model.eval()
        
        skip_batch=0
        
        if is_train:
            inference_model.train()
        
        batch_size = batch_size//2
             
        ##### load memguard's prediciton logits on known members and non-members 
        if(eval_set_identifier_4_memguard == 'train'):
            tr_outputs =  shadow_train_performance[0][: num_sample]
            te_outputs = shadow_test_performance[0][:num_sample]
            tr_labels = shadow_train_performance[1][: num_sample]
            te_labels = shadow_test_performance[1][:num_sample]

        elif(eval_set_identifier_4_memguard == 'val'):
            tr_outputs =  shadow_train_performance[0][num_sample:]
            te_outputs =  shadow_test_performance[0][num_sample:]
            tr_labels = shadow_train_performance[1][ num_sample:]
            te_labels = shadow_test_performance[1][num_sample:]

        elif(eval_set_identifier_4_memguard=='test'):
            tr_outputs = target_train_performance[0]
            te_outputs = target_test_performance[0]
            tr_labels = target_train_performance[1]
            te_labels = target_test_performance[1]

        all_tr_outputs = tr_outputs
        all_te_outputs = te_outputs
        all_tr_labels = tr_labels
        all_te_labels = te_labels

        len_t = len(all_tr_outputs)//batch_size
        if len(all_tr_outputs)%batch_size:
            len_t += 1

        for ind in range(skip_batch, len_t):
            if ind >= skip_batch+num_batchs:
                break
            
            pred_outputs = np.r_[ all_tr_outputs[ind*batch_size:(ind+1)*batch_size], all_te_outputs[ind*batch_size:(ind+1)*batch_size] ] 
            pred_outputs = torch.from_numpy(pred_outputs).type(torch.FloatTensor).cuda()
            tr_target = torch.from_numpy(all_tr_labels[ind*batch_size:(ind+1)*batch_size]).type(torch.LongTensor).cuda()
            te_target = torch.from_numpy(all_te_labels[ind*batch_size:(ind+1)*batch_size]).type(torch.LongTensor).cuda()
            v_tr_target = torch.autograd.Variable(tr_target)
            v_te_target = torch.autograd.Variable(te_target)
            model_input= torch.cat((v_tr_target,v_te_target))
            infer_input= torch.cat((v_tr_target,v_te_target))

            one_hot_tr = torch.from_numpy(np.zeros(pred_outputs.size())).cuda().type(torch.cuda.FloatTensor)
            target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.cuda.LongTensor).view([-1,1]).data,1)

            infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

            loss_= classifier_criterion_noreduct(pred_outputs, infer_input).view([-1,1])

            preds = torch.autograd.Variable(torch.from_numpy(pred_outputs.data.cpu().numpy()).cuda())

            member_output = inference_model(pred_outputs, infer_input_one_hot, loss_)
            # print(member_output) 


            is_member_labels = torch.from_numpy(np.reshape(np.concatenate((np.zeros(v_tr_target.size(0)),np.ones(v_te_target.size(0)))),[-1,1])).cuda()
            
            v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.cuda.FloatTensor)

            loss = criterion_attck(member_output, v_is_member_labels)

            # measure accuracy and record loss
            prec1=np.mean((member_output.data.cpu().numpy() >0.5)==v_is_member_labels.data.cpu().numpy())
            losses.update(loss.item(), model_input.size(0))
            top1.update(prec1, model_input.size(0))


            predicted_member_labels = member_output.data.cpu().numpy() >0.5
            actual_member_labels = v_is_member_labels.data.cpu().numpy()


            accuracy = tf.keras.metrics.Accuracy()
            precision = tf.keras.metrics.Precision()
            recall = tf.keras.metrics.Recall()
            accuracy.update_state(actual_member_labels, predicted_member_labels)
            precision.update_state(actual_member_labels, predicted_member_labels)
            recall.update_state(actual_member_labels, predicted_member_labels)

            if( precision.result() + recall.result() != 0 ):
                F1_Score = 2 * (precision.result() * recall.result()) / (precision.result() + recall.result())
            else:
                F1_Score = 0

            acys.update(accuracy.result(),  model_input.size(0))
            precisions.update(precision.result(),  model_input.size(0))
            recalls.update(recall.result(),  model_input.size(0))
            F1_Scores.update(F1_Score,  model_input.size(0))


            # compute gradient and do SGD step
            optimizer.zero_grad()
            if is_train:
                loss.backward()
                optimizer.step()

            # plot progress
            if False and ind%10==0:
                print  ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                        batch=ind ,
                        size=len_t,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        ))

            if(ind==0):
                raw_pred_socre = member_output.data.cpu().numpy()
                true_label = v_is_member_labels.data.cpu().numpy()
            else:
                raw_pred_socre = np.r_[ raw_pred_socre,  member_output.data.cpu().numpy()]
                true_label = np.r_[ true_label,  v_is_member_labels.data.cpu().numpy()] 

        return (losses.avg, top1.avg, acys.avg, precisions.avg, recalls.avg, F1_Scores.avg, raw_pred_socre, true_label)


    user_lr=0.0005
    at_lr=0.0005
    at_schedule=[100]
    at_gamma=0.1
    n_classes=num_classes
    # criterion_classifier = nn.CrossEntropyLoss(reduction='none')
    criterion_classifier = cross_entropy_loss
    attack_criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    best_at_val_acc=0
    best_at_test_acc=0
    attack_epochs=200 
    attack_model = InferenceAttack_BB(n_classes)
    attack_model = attack_model.cuda()
    attack_optimizer = optim.Adam(attack_model.parameters(),lr=at_lr)
    # BATCH_SIZE = 256

    save_tag = config.attack.save_tag

    max_acc=0
    max_auc=0
    for epoch in range(attack_epochs):
        if epoch in at_schedule:
            for param_group in attack_optimizer.param_groups:
                param_group['lr'] *= at_gamma
                # print('Epoch %d Local lr %f'%(epoch,param_group['lr']))
 

        at_loss, at_acc, at_acy, at_precision, at_recall, at_f1, _, _ = attack_bb( attack_model, criterion, criterion_classifier, attack_criterion, 
                                    attack_optimizer, epoch, is_train=True, batch_size=BATCH_SIZE, eval_set_identifier_4_memguard='train' )

        # at_val_loss, at_val_acc, at_val_acy, at_val_precision, at_val_recall, at_val_f1, _, _  = attack_bb( attack_model, criterion, criterion_classifier, attack_criterion, 
        #                                     attack_optimizer, epoch,  is_train=False, batch_size=BATCH_SIZE, eval_set_identifier_4_memguard='val' )

        # is_best = at_val_acc > best_at_val_acc

        # if is_best:
        at_test_loss, at_test_acc, at_best_acy, at_best_precision, at_best_recall, at_best_f1, y_score, y_true  = attack_bb( attack_model, criterion, criterion_classifier, attack_criterion, 
                                                       attack_optimizer, epoch,  is_train=False, batch_size=BATCH_SIZE, eval_set_identifier_4_memguard='test' )


        if at_test_acc>best_at_test_acc:
            y_true_best, y_score_best = y_true, y_score
        best_at_test_acc = max(best_at_test_acc, at_test_acc)

        if(epoch == attack_epochs-1):
            print()
            print("\t===>   NN-based attack ", config.attack.path)

        # if( (epoch+1)%5==0 ):
        #     #print(' Epoch %d | current stats acy: %.4f precision: %.4f recall: %.4f F1_Score: %.4f | best test stats: %.4f precision: %.4f recall: %.4f F1_Score: %.4f '\
        #     #            %(epoch, at_val_acc, at_val_precision, at_val_recall, at_val_f1,\
        #     #                 best_at_test_acc, at_best_precision, at_best_recall, at_best_f1) , flush=True)
        #     print(' Epoch %d '%epoch)
        #     atk_auc, atk_acc = get_tpr(y_true, y_score, config.attack.fpr_threshold, 'nn-based-%s.npy'%save_tag)

    #np.save( os.path.join(output_save_path, 'nn-based-%s.npy'%save_tag), np.r_[y_true, y_score] )
    # atk_auc, atk_acc = get_tpr(y_true_best, y_score_best, config.attack.fpr_threshold, 'nn-based-%s.npy'%save_tag)
    atk_auc, atk_acc = get_tpr(y_true, y_score, config.attack.fpr_threshold, 'nn-based-%s.npy'%save_tag)
    attack_aucs.append(atk_auc)
    attack_accs.append(atk_acc)

if(config.attack.attack == 'entropy' or config.attack.attack == 'all'):
    # print(target_train_performance[0].shape)

    #print("\t===> correctness-based, entropy-based, m-entropy-based-, confidence-based attacks ", config.attack.path)
    MIA = black_box_benchmarks(shadow_train_performance,shadow_test_performance,
                         target_train_performance,target_test_performance,num_classes=num_classes)
    MIA._mem_inf_benchmarks()

    save_tag = config.attack.save_tag

    y_score = np.r_[ MIA.s_tr_m_entr, MIA.t_tr_m_entr, MIA.s_te_m_entr, MIA.t_te_m_entr ]
    y_true = np.r_[ np.ones( len(MIA.t_tr_labels) + len(MIA.s_tr_labels)  ), np.zeros( len(MIA.t_te_labels) + len(MIA.s_te_labels) ) ]

    y_score *= -1 # roc default > takes positive label; but we want < takes positive label
    #np.save( os.path.join(output_save_path, 'm-entropy-based-%s.npy'%save_tag), np.r_[y_true, y_score] ) 
    atk_auc, atk_acc = get_tpr(y_true, y_score, config.attack.fpr_threshold, 'm-entropy-based-%s.npy'%save_tag)
    attack_aucs.append(atk_auc)
    attack_accs.append(atk_acc)

    y_score = np.r_[ MIA.s_tr_conf, MIA.t_tr_conf, MIA.s_te_conf, MIA.t_te_conf ] 
    #np.save( os.path.join(output_save_path,  'confidence-based-%s.npy'%save_tag), np.r_[y_true, y_score] ) 
    atk_auc, atk_acc = get_tpr(y_true, y_score, config.attack.fpr_threshold, 'confidence-based-%s.npy'%save_tag)
    attack_aucs.append(atk_auc)
    attack_accs.append(atk_acc)

    y_score = np.r_[ MIA.s_tr_entr, MIA.t_tr_entr,  MIA.s_te_entr, MIA.t_te_entr ]
    y_score *= -1 # roc default > takes positive label; but we want < takes positive label 
    #np.save( os.path.join(output_save_path,  'entropy-based-%s.npy'%save_tag), np.r_[y_true, y_score] )
    atk_auc, atk_acc = get_tpr(y_true, y_score, config.attack.fpr_threshold, 'entropy-based-%s.npy'%save_tag)
    attack_aucs.append(atk_auc)
    attack_accs.append(atk_acc)

    known_tr_loss = memguard_get_pred_loss(shadow_train_performance[0],shadow_train_performance[1])
    known_te_loss = memguard_get_pred_loss(shadow_test_performance[0],shadow_test_performance[1])

    tr_loss = memguard_get_pred_loss(target_train_performance[0],target_train_performance[1])
    te_loss = memguard_get_pred_loss(target_test_performance[0],target_test_performance[1])

  
    y_true = np.r_[ np.ones(len(known_tr_loss)+len(tr_loss)) , np.zeros(len(known_te_loss) +len(te_loss)) ]
    y_score = np.r_[ known_tr_loss, tr_loss, known_te_loss, te_loss ]

    y_score *= -1 # roc default > takes positive label; but we want < takes positive label

    save_tag = config.attack.save_tag

    #np.save( os.path.join(output_save_path, 'loss-based-%s.npy'%save_tag), np.r_[y_true, y_score] ) 
    atk_auc, atk_acc = get_tpr(y_true, y_score, config.attack.fpr_threshold, 'loss-based-%s.npy'%save_tag)
    attack_aucs.append(atk_auc)
    attack_accs.append(atk_acc)

    print(f'Best Attack AUC:{max(attack_aucs): .4f}')
    print(f'Best Attack ACC:{max(attack_accs): .4f}')
    '''
    print("\t===> loss-based attack ", args.path)
    m_pred = loss_threshold_attack(best_model, query_model, torch.cat( (mia_test_members_data_tensor, mia_test_nonmembers_data_tensor), 0), 
                                    torch.cat( (mia_test_members_label_tensor, mia_test_nonmembers_label_tensor), 0), 
                                    tr_members, tr_members_y,
                                    non_Mem_Generator,  10)

    m_true = np.r_[ np.ones( len(mia_test_members_data_tensor) ), np.zeros( len(mia_test_nonmembers_data_tensor) )  ] 
    pred_label = m_pred
    eval_label = m_true
    print("\tAccuracy: %.4f | Precision %.4f | Recall %.4f | f1_score %.4f" % ( accuracy_score(eval_label, pred_label), precision_score(eval_label,pred_label),\
                                recall_score(eval_label,pred_label), f1_score(eval_label,pred_label)))

    '''

if(config.attack.getModelAcy):
    print('{} | train acc {:.4f} | test acc {:.4f}'.format(config.attack.path, check_train_acc,check_test_acc), flush=True)
end = time.time()
print('running time:', end-start)
     