import sys
import argparse
import os 
parser = argparse.ArgumentParser()
parser.add_argument('--train_org', type=int, default=0, help='flag for training teacher model')
parser.add_argument('--train_selena', type=int, default=0, help='flag for performing knowledge distillation')
parser.add_argument('--idx_pre', type=int, default=0, help='flag for prepare idx')
parser.add_argument('--K', type=int, default=25, help='num of teacher models')
parser.add_argument('--L', type=int, default=10, help='num of models that are not trained on each samples')
parser.add_argument('--config', type=str, default='./configs/selena.yml')
args = parser.parse_args()
sys.path.insert(0,'./util/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import sys
import os 
from util.densenet import densenet
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model_factory import *


config = parse_config(args.config)
args.model = config.trainer.classifier
model_checkpoint_dir=f'./final-all-models/{args.model}'
args.model_save_tag = config.trainer.save_tag

float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
org_model_checkpoint_dir=os.path.join(model_checkpoint_dir,'./selena')
if(not os.path.exists(org_model_checkpoint_dir)):
    os.makedirs(org_model_checkpoint_dir,exist_ok=True)

is_train_org = args.train_org
is_train_ref = args.train_selena
BATCH_SIZE =config.trainer.batch_size

num_epochs=config.trainer.classifier_epochs
distil_epochs=config.trainer.ditill_epochs

import random
random.seed(1)
# prepare test data parts
trainset, testset, privateset,refset,trainset_origin,privateset_origin, num_classes = get_dataset()

print("loading data")

private_dataloader = torch.utils.data.DataLoader(privateset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

print('tr len %d | val data len %d'%
      (len(privateset),len(testset)), flush=True)


private_data_len = len(privateset)
all_indices=np.arange(len(trainset))


'''
For each training sample, we randomly select L models from the K teacher models and store them in ``non_model_indices_for_each_sample''
    These L models are the models that are not trained on the specific training samples
For each teacher model, we also generate a list ``sub_model_indices'' to its reference set (on which the teacher model should be trained)
'''
np.random.seed(0)
K = args.K
L = args.L
sub_model_indices_file_prefix = org_model_checkpoint_dir+'/cifar_selena_submodel_indices'
non_model_indices_file = org_model_checkpoint_dir +'/non_model_indices_for_each_sample.pkl'
sub_model_indices = [[] for _ in range(K)]
non_model_indices_for_each_sample = np.zeros((private_data_len, L))
if not os.path.isfile( sub_model_indices_file_prefix + "_0.pkl" ):
    # partition this into K lists
    training_indices = all_indices[:private_data_len]  
    for cnt, each_ind in enumerate(training_indices):
        non_model_indices = np.random.choice(K, L, replace=False) # out of K teacher models, L of them will not be trained on the current sample
        non_model_indices_for_each_sample[cnt] = non_model_indices # L indices for each sample 
 
        for i in range(K):
            if(i not in non_model_indices):
                # the current index will be stored for the i_th teacher model
                # these are the ``reference set'' that will be used for training the sub model
                sub_model_indices[i].append(each_ind)

    sub_model_indices = np.asarray(sub_model_indices)
    non_model_indices_for_each_sample = np.asarray(non_model_indices_for_each_sample)
    for i in range(K): 
        pickle.dump(sub_model_indices[i] , open(sub_model_indices_file_prefix + "_%s.pkl"%str(i), 'wb') )
    pickle.dump(non_model_indices_for_each_sample, open(non_model_indices_file, 'wb'))

else:
    for i in range(K):
        # indices to the ``reference set'' for the teacher model
        sub_model_indices[i] = pickle.load( open(sub_model_indices_file_prefix + "_%s.pkl"%str(i), 'rb') )  
    # indices to the ``teacher model'' for each sample, on which the indexed teacher model is ``not'' trained
    non_model_indices_for_each_sample = pickle.load( open(non_model_indices_file, 'rb') ) 


def train_pub(train_loader, model, t_softmax, optimizer, num_batchs=999999, batch_size=16, alpha=1):
    # switch to train mode
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    true_criterion=nn.CrossEntropyLoss()

    
    for ind,(inputs,targets,true_targets) in enumerate(train_loader):
        if ind > num_batchs:
            break
        inputs, targets, true_targets = inputs.to(device), targets.to(device), true_targets.to(device)
        inputs, targets, true_targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets), torch.autograd.Variable(true_targets)
        outputs  = model(inputs)
        loss = alpha*F.kl_div(F.log_softmax(outputs/t_softmax, dim=1), F.softmax(targets/t_softmax, dim=1)) + (1-alpha)*true_criterion(outputs,true_targets)
        losses.update(loss.item(), inputs.size(0)) 
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg)


def train(train_loader,model,criterion,optimizer,epoch,use_cuda,num_batchs=999999,batch_size=32):
    # switch to train mode
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for ind,(inputs,targets) in enumerate(train_loader): 

        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs  = model(inputs)
        loss = criterion(outputs, targets)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg, top1.avg)


import time
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


elapsed_time = 0
criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
if(is_train_org): 
    for sub in range(K):
        model=create_model(model_name = args.model, num_classes=num_classes)
        model=model.to(device)
        # optimizer = optim.SGD(model.parameters(), lr=user_lr, momentum=0.9, weight_decay=1e-4)  
        optimizer, scheduler = parser_opt_scheduler(model,config)
        best_val_acc=0
        best_test_acc=0        
        # derive the ``reference set'' for training the teacher model
        sub_trainset =  torch.utils.data.Subset(trainset, sub_model_indices[sub])
        print(len(sub_trainset))
        sub_train_loader = torch.utils.data.DataLoader(sub_trainset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(num_epochs):
            start_time = time.time()
            # adjust_learning_rate(optimizer, epoch)
            train_loss, train_acc = train(sub_train_loader, model, criterion, optimizer, epoch, use_cuda, batch_size= BATCH_SIZE)

            val_loss, val_acc = test(test_dataloader, model, criterion, use_cuda, device=device)

            is_best = val_acc > best_val_acc
            best_val_acc=max(val_acc, best_val_acc)

            save_checkpoint_global(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_acc': best_val_acc,
                    'optimizer': optimizer.state_dict(),
                },
                is_best,
                checkpoint=org_model_checkpoint_dir,
                filename='unprotected_model_sub_%s.pth.tar'%str(sub),
                best_filename='unprotected_model_best_sub_%s.pth.tar'%str(sub),
            )

            print('%d sub model |  epoch %d | tr acc %.2f loss %.2f | val acc %.2f loss %.2f  | best val acc %.2f '
                  %(sub, epoch, train_acc, train_loss, val_acc, val_loss, best_val_acc), flush=True)

            epoch_time = time.time() - start_time
            elapsed_time += epoch_time
            print('| Elapsed time : %d hr, %02d min, %02d sec'  %(get_hms(elapsed_time)))
            if scheduler is not None:
                scheduler.step()


if(is_train_ref):

    # distil the knowledge of the unprotected model in the ref data 
    criterion=nn.CrossEntropyLoss()
    sub_models = [[] for _ in range(K)]
    for i in range(K):
        best_model=create_model(model_name = args.model, num_classes=num_classes).to(device)
        resume_best=os.path.join( org_model_checkpoint_dir,'unprotected_model_best_sub_%s.pth.tar'%str(i))
        assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model %s'%resume_best
        checkpoint = os.path.dirname(resume_best)
        checkpoint = torch.load(resume_best)
        best_model.load_state_dict(checkpoint['state_dict'])
        _,best_val = test(test_dataloader, best_model, criterion, use_cuda)
        _,best_train = test(private_dataloader, best_model, criterion, use_cuda)
        print(' %s sub model: train acc %.4f val acc %.4f'%(resume_best, best_train, best_val), flush=True)
        sub_models[i] = best_model
    training_indices = all_indices[:private_data_len]  

    private_dataloader_1 = torch.utils.data.DataLoader(privateset, batch_size=1, shuffle=False)

    all_outputs_file = org_model_checkpoint_dir+'/all_outputs.pkl'
    if not os.path.isfile( all_outputs_file ):
        all_outputs=[]
        non_model_indices_for_each_sample = non_model_indices_for_each_sample.astype(int)
        for cnt, (inputs,targets) in enumerate(private_dataloader_1):  
            inputs = inputs.to(device)
            first = True
            # adaptive inference on the sub model, each model only predicts on the samples that were NOT used for training
            for i, model_index in enumerate(non_model_indices_for_each_sample[cnt]): 
                sub_model = sub_models[model_index]
                outputs = sub_model(inputs)
                if(first):
                    outs = outputs 
                    first=False
                else:
                    outs = torch.cat( (outs, outputs), 0)
            # aggregate all the scores to generate soft labels for distillation
            all_outputs.append( torch.mean(outs, dim=0).cpu().detach().numpy() )

        all_outputs = np.asarray(all_outputs)
        pickle.dump(all_outputs, open(all_outputs_file, 'wb'))
    else:
        all_outputs = pickle.load( open(all_outputs_file, 'rb') ) 




    distil_label_tensor=(torch.from_numpy(all_outputs).type(torch.FloatTensor))
    # ditillset = Cifardata(privateset.data,distil_label_tensor)
    ditillset = Hampdata(privateset,distil_label_tensor)
    distill_loader = torch.utils.data.DataLoader(ditillset, batch_size=BATCH_SIZE, shuffle=False)

    # train final protected model via knowledge distillation
    distil_model=create_model(model_name = args.model, num_classes=num_classes).to(device)
    distil_test_criterion=nn.CrossEntropyLoss()
    distil_best_acc=0
    best_distil_test_acc=0
    gamma=.1
    t_softmax=1


    for epoch in range(distil_epochs):
        start_time = time.time()
        # if epoch in distil_schedule:
        #     distil_lr *= gamma
        #     print('----> Epoch %d distillation lr %f'%(epoch,distil_lr))

        # distil_optimizer=optim.SGD(distil_model.parameters(), lr=distil_lr, momentum=0.99, weight_decay=1e-5)
        distil_optimizer, distil_scheduler = parser_opt_scheduler(distil_model,config)
        # distil_optimizer=optim.Adam(distil_model.parameters(), lr=0.0001, betas=[0.9, 0.999], weight_decay=1e-6)
        distil_tr_loss = train_pub(distill_loader, distil_model, t_softmax,
                                   distil_optimizer, batch_size=BATCH_SIZE, alpha=1)

        tr_loss,tr_acc = test(private_dataloader, distil_model, distil_test_criterion, use_cuda, device=device)
        
        val_loss,val_acc = test(test_dataloader, distil_model, distil_test_criterion, use_cuda, device=device)

        distil_is_best = val_acc > distil_best_acc
        distil_best_acc=max(val_acc, distil_best_acc)
      
        save_checkpoint_global(
            {
                'epoch': epoch,
                'state_dict': distil_model.state_dict(),
                'best_acc': distil_best_acc,
                'optimizer': distil_optimizer.state_dict(),
            },
            distil_is_best,
            checkpoint=org_model_checkpoint_dir,
            filename='protected_model-%s.pth.tar'%args.model_save_tag,
            best_filename=f'{args.model_save_tag}.pth.tar',
        )
        print('epoch %d | distil loss %.4f | tr loss %.4f tr acc %.4f | val loss %.4f val acc %.4f | best val acc %.4f | best test acc %.4f'%(epoch,distil_tr_loss,tr_loss,tr_acc,val_loss,val_acc,distil_best_acc,best_distil_test_acc),flush=True)

        elapsed_time += time.time() - start_time
        print('| Elapsed time : %d hr, %02d min, %02d sec'  %(get_hms(elapsed_time)))
        if distil_scheduler is not None:
            distil_scheduler.step()


if args.idx_pre:
    print('done')
    sys.exit()
criterion=nn.CrossEntropyLoss()
best_model=create_model(model_name = args.model, num_classes=num_classes).to(device)
resume_best= os.path.join(org_model_checkpoint_dir, '../',f'{args.model_save_tag}.pth.tar')
print(resume_best)
assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model'
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best)
best_model.load_state_dict(checkpoint['state_dict'])
_,best_val = test(test_dataloader, best_model, criterion, use_cuda, device=device)
_,best_train = test(private_dataloader, best_model, criterion, use_cuda, device=device)
print('\t===> SELENA %s | train acc %.4f | val acc %.4f'%(resume_best, best_train, best_val), flush=True)

