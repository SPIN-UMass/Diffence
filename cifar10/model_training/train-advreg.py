import sys
import argparse
import os
parser = argparse.ArgumentParser()   
parser.add_argument('--train_org', type=int, default=1)
parser.add_argument('--alpha', type=int, default=0)
parser.add_argument('--config', type=str, default='./configs/advreg.yml')
args = parser.parse_args()
sys.path.insert(0,'./util/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import torch.optim as optim
import argparse
import os
import shutil
import time
import random
import torch.nn.functional as F
from densenet_advreg import densenet,AdvRegWarper
import torch
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from model_factory import *


config = parse_config(args.config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.model = config.trainer.classifier
org_model_checkpoint_dir=f'./final-all-models/{args.model}'
args.model_save_tag = config.trainer.save_tag
if args.alpha==0:
    args.alpha = config.trainer.alpha

random.seed(1)
# prepare test data parts
trainset, testset, privateset,refset, trainset_origin, privateset_origin, num_classes = get_dataset()
BATCH_SIZE =config.trainer.batch_size
num_epochs=config.trainer.classifier_epochs
# args.alpha = config.trainer.alpha

print("loading data")
private_dataloader = torch.utils.data.DataLoader(privateset, batch_size=BATCH_SIZE, shuffle=True)
private_dataloader_origin = torch.utils.data.DataLoader(privateset_origin, batch_size=BATCH_SIZE, shuffle=True)
ref_dataloader = torch.utils.data.DataLoader(refset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

print('tr len %d |ref data len %d |val data len %d'%
      (len(privateset),len(refset),len(testset)), flush=True)
# checkpoint_dir='./advreg-model'
checkpoint_dir=f'./final-all-models/{args.model}'
if(not os.path.exists(checkpoint_dir)):
    os.mkdir(checkpoint_dir)

import time
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

global elapsed_time
elapsed_time = 0
start_time = time.time()

def advtune_defense(num_epochs=50, use_cuda=True,batch_size=64,alpha=0,lr=0.0005,schedule=[25,80],gamma=0.1,tr_epochs=100,at_lr=0.0001,at_schedule=[100],at_gamma=0.5,at_epochs=200,n_classes=10):
 
    global elapsed_time
    ############################################################ private training ############################################################
    print('Training using adversarial tuning...')
    model=create_model(model_name = args.model, num_classes=num_classes)
    model = AdvRegWarper(model)
    model=model.to(device)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr = 0.001, betas = (0.9, 0.999), amsgrad = True, weight_decay=1e-6)
    criterion=nn.CrossEntropyLoss()
    optimizer, scheduler = parser_opt_scheduler(model,config)

    attack_model=InferenceAttack_HZ(n_classes)
    
    attack_optimizer=optim.Adam(attack_model.parameters(),lr=at_lr)
    attack_criterion=nn.MSELoss()

    if(use_cuda):
        attack_model=attack_model.cuda()
        model=model.cuda()

    best_acc=0
    best_test_acc=0    
    for epoch in range(num_epochs):
        start_time = time.time()

        # decay the lr at certain epoches in schedule
        # adjust_learning_rate(optimizer, epoch) 
        if scheduler is not None:
            scheduler.step()

        c_batches = len(private_dataloader)
        if epoch == 0:
            print('----> NORMAL TRAINING MODE: c_batches %d '%(c_batches), flush=True)


            train_loss, train_acc = train(private_dataloader,
                                              model,criterion,optimizer,epoch,use_cuda,debug_='MEDIUM')    
            test_loss, test_acc = test(test_dataloader,model,criterion,use_cuda, batch_size=batch_size)    
            for i in range(5):
                at_loss, at_acc = train_attack(private_dataloader_origin,ref_dataloader,model,attack_model,criterion,
                                               attack_criterion,optimizer,attack_optimizer,epoch,use_cuda, batch_size=batch_size,debug_='MEDIUM')    

            print('Initial test acc {} train att acc {}'.format(test_acc, at_acc), flush=True)

        else:
            
            # for e_num in schedule:
            #     if e_num==epoch:
            #         for param_group in optimizer.param_groups:
            #             param_group['lr'] *= gamma
            #             print('Epoch %d lr %f'%(epoch,param_group['lr']))

            att_accs =[]

            rounds=(c_batches//2)

            for i in range(rounds):

                at_loss, at_acc = train_attack(private_dataloader_origin, ref_dataloader,
                                               model,attack_model,criterion,attack_criterion,optimizer,
                                               attack_optimizer,epoch,use_cuda,52//2,None,batch_size=batch_size)

                att_accs.append(at_acc)

                tr_loss, tr_acc = train_privatly(private_dataloader,model,
                                                 attack_model,criterion,optimizer,epoch,use_cuda,
                                                 2,None,alpha=alpha,batch_size=batch_size)

            train_loss,train_acc = test(private_dataloader_origin,model,criterion,use_cuda)
            val_loss, val_acc = test(test_dataloader,model,criterion,use_cuda)
            is_best = (val_acc > best_acc)

            best_acc=max(val_acc, best_acc)

            at_val_loss, at_val_acc = test_attack(private_dataloader_origin,ref_dataloader,
                                                     model,attack_model,criterion,attack_criterion,
                                                     optimizer,attack_optimizer,epoch,use_cuda,debug_='MEDIUM')
            
            att_epoch_acc = np.mean(att_accs)
            
            save_checkpoint_global(
               {
                   'epoch': epoch,
                   'state_dict': model.model.state_dict(),
                   'acc': val_acc,
                   'best_acc': best_acc,
                   'optimizer': optimizer.state_dict(),
               },
               is_best,
               checkpoint=checkpoint_dir,
               filename='protected_model-%s.pth.tar'%args.model_save_tag,
               best_filename=f'{args.model_save_tag}.pth.tar',
            )
          
            print('epoch %d | tr_acc %.2f | val acc %.2f | best val acc %.2f | best te acc %.2f | attack avg acc %.2f | attack val acc %.2f'%(epoch,train_acc,val_acc,best_acc,best_test_acc,att_epoch_acc,at_val_acc), flush=True)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d hr, %02d min, %02d sec'  %(get_hms(elapsed_time)))
    ############################################################ private training ############################################################

if args.train_org:
    advtune_defense(alpha=args.alpha, batch_size=BATCH_SIZE, num_epochs=num_epochs, use_cuda=True,n_classes=num_classes)
 
best_model=create_model(model_name = args.model, num_classes=num_classes)
# best_model = AdvRegWarper(best_model)
criterion=nn.CrossEntropyLoss()
use_cuda = True
resume_best= os.path.join(checkpoint_dir, f'{args.model_save_tag}.pth.tar')

assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model'
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best)
best_model.load_state_dict(checkpoint['state_dict'])

best_model = best_model.to(device)
_,best_val = test(test_dataloader, best_model, criterion, use_cuda, device=device)
_,best_train = test(private_dataloader_origin, best_model, criterion, use_cuda, device=device)
print('\t===> AdvReg model %s | train acc %.4f | val acc %.4f'%(resume_best, best_train, best_val), flush=True)


