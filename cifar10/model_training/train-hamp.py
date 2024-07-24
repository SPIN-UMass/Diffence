import sys
sys.path.insert(0,'./util/')
import os
import argparse
parser = argparse.ArgumentParser()   
parser.add_argument('--isTrain', type=int, default = 1)
parser.add_argument('--scan_para', type=int, default = 0)
parser.add_argument('--entropy_percentile', type=float, default=0, help="gamma parameter in hamp")
parser.add_argument('--entropy_penalty', type=int, default=1, help='flag to indicate whether to use regularization or not')
parser.add_argument('--alpha', type=float, default=0, help='alpha parameter in hamp')
parser.add_argument('--config', type=str, default='./configs/hamp.yml')
args = parser.parse_args()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import sys
import os 
from torch.nn import functional as F
from util.densenet import densenet
from torch.distributions import Categorical
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
if args.scan_para==0:
    model_checkpoint_dir=f'./final-all-models/{args.model}/'
else:
    model_checkpoint_dir=f'./final-all-models/{args.model}/hamp'

if args.entropy_percentile==0:
    args.entropy_percentile = config.trainer.entropy_percentile
if args.alpha==0:
    args.alpha = config.trainer.alpha

args.model_save_tag = config.trainer.save_tag

seed = 1
np.random.seed(seed)
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('entropy_percentile:', args.entropy_percentile)  # lower means higher utility
print('alpha:', args.alpha)


# new_model_checkpoint_dir='./hamp-entropyPert-{}-trainSize-{}-w-entropyReg-alpha-{}'.format(str(args.entropy_percentile).replace('.', '') , str(args.train_size), str(args.alpha).replace('.', '-')  )
new_model_checkpoint_dir = model_checkpoint_dir
if(not os.path.exists(new_model_checkpoint_dir)):
    try:
        os.mkdir(new_model_checkpoint_dir)
    except:
        print('folder exists already')

is_train_ref = args.isTrain 
BATCH_SIZE =config.trainer.batch_size

print("loading data")
trainset, testset, privateset,refset,trainset_origin,privateset_origin, num_classes = get_dataset()
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

private_data_len= len(privateset)
ref_data_len = len(refset) 
num_epochs=config.trainer.classifier_epochs

print('tr len %d | val data len %d'%
      (len(privateset),len(testset)), flush=True)


def train_pub(train_loader, model, t_softmax, optimizer, num_batchs=999999, batch_size=16, alpha=0.05):
    # switch to train mode
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    true_criterion=nn.CrossEntropyLoss()

    for ind, (inputs,targets,true_targets) in enumerate(train_loader):
        if ind > num_batchs:
            break

        inputs, targets, true_targets = inputs.to(device), targets.to(device), true_targets.to(device)
        inputs, targets, true_targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets), torch.autograd.Variable(true_targets)
        outputs  = model(inputs)
        if(not args.entropy_penalty): 
            loss = F.kl_div(F.log_softmax(outputs, dim=1), targets )  
        else:
            entropy = Categorical(probs = F.softmax(outputs, dim=1)).entropy()
            loss1 = F.kl_div(F.log_softmax(outputs, dim=1), targets )   
            loss2 = -1 * alpha * torch.mean(entropy)
            loss = loss1 + loss2
        losses.update(loss.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg)


def entropy(preds, axis=0):
    logp = np.log(preds)
    entropy = np.sum( -preds * logp , axis=axis ) 
    return entropy

def get_top1(num_classes, entropy_threshold, reduced_prob=0.01):
    # reduced_prob : for reducing top-1 class's probability
    true_target = 1
    preds = np.zeros(num_classes)
    preds[true_target]  = 1.
    while(True):
        preds[true_target] -= reduced_prob
        preds[:true_target] += reduced_prob/(num_classes-1) 
        preds[true_target+1:] += reduced_prob/(num_classes-1) 
        if(entropy(preds) >= entropy_threshold):
            break
    return preds[true_target], preds[true_target+1]

def get_soft_labels(train_label, num_classes, top1, uniform_non_top1):
    new_soft_label = np.zeros( (train_label.shape[0], num_classes) ) 
    for i in range( train_label.shape[0] ):
        new_soft_label[i][train_label[i]] = top1
        new_soft_label[i][:train_label[i]] = uniform_non_top1
        new_soft_label[i][train_label[i]+1:] = uniform_non_top1
    print( new_soft_label[0], train_label[0], np.argmax(new_soft_label[0]) )
    return new_soft_label

use_cuda = torch.cuda.is_available()
num_class = num_classes
preds = np.ones(num_class)
preds /= float(num_class)   
highest_entropy = entropy(preds) 
# assign uniform class prob for all the non-top-1 classes
top1, uniform_non_top1 = get_top1(num_class, highest_entropy*args.entropy_percentile)
print("Highest entropy {:.4f} | entropy_percentile {:.4f} | entropy threshold {:.4f}".format(highest_entropy , args.entropy_percentile, highest_entropy*args.entropy_percentile))


private_label_modified = get_soft_labels(privateset.labels, num_class, top1, uniform_non_top1)
private_label_modified_tensor=torch.from_numpy(private_label_modified).type(torch.FloatTensor)
hampset = Hampdata(privateset,private_label_modified_tensor)
private_dataloader = torch.utils.data.DataLoader(hampset, batch_size=BATCH_SIZE, shuffle=True)
private_dataloader_origin = torch.utils.data.DataLoader(privateset, batch_size=BATCH_SIZE, shuffle=True)

distil_epochs= num_epochs
distil_model=create_model(model_name = args.model, num_classes=num_classes).to(device)
distil_test_criterion=nn.CrossEntropyLoss()
# distil_optimizer=optim.SGD(distil_model.parameters(), lr=distil_lr, momentum=0.99, weight_decay=1e-5)
# distil_optimizer= optim.Adam(distil_model.parameters(), lr = 0.001, betas = (0.9, 0.999), amsgrad = True, weight_decay=1e-6)

distil_optimizer, scheduler = parser_opt_scheduler(distil_model, config)


distil_best_acc=0
best_distil_test_acc=0
t_softmax=1
best_train_acc = 0.

import time
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s
elapsed_time = 0

if args.scan_para==0:
    best_filename = f'{args.model_save_tag}.pth.tar'
else:
    best_filename = f'{args.model_save_tag}_{args.entropy_percentile}_{args.alpha}.pth.tar'
if(is_train_ref):
    for epoch in range(distil_epochs): 
        start_time = time.time()
        # adjust_learning_rate(distil_optimizer, epoch) 
        distil_tr_loss = train_pub(private_dataloader, distil_model, t_softmax,
                                   distil_optimizer, batch_size=BATCH_SIZE, alpha=args.alpha)
        tr_loss,tr_acc = test(private_dataloader_origin, distil_model, distil_test_criterion, use_cuda, device=device)
        val_loss,val_acc = test(test_dataloader, distil_model, distil_test_criterion, use_cuda, device=device)
        # the validation acy needs to increase by at least 1% 
        distil_is_best = val_acc > distil_best_acc
        distil_best_acc=max(val_acc, distil_best_acc)
        if distil_is_best:
            best_train_acc = tr_acc

        save_checkpoint_global(
            {
                'epoch': epoch,
                'state_dict': distil_model.state_dict(),
                'best_acc': distil_best_acc,
                'optimizer': distil_optimizer.state_dict(),
            },
            distil_is_best,
            checkpoint=new_model_checkpoint_dir,
            filename=f'protected_model-{args.model_save_tag}_{args.entropy_percentile}_{args.alpha}.pth.tar',
            best_filename=best_filename,
        )

        print('lr %.5f | alpha %.3f epoch %d | loss %.4f | tr acc %.4f | val acc %.4f || best tr acc %.4f| best val acc %.4f'%(distil_optimizer.param_groups[0]['lr'], args.alpha, epoch,distil_tr_loss,tr_acc,val_acc, best_train_acc,distil_best_acc), flush=True)


        elapsed_time += time.time() - start_time
        print('| Elapsed time : %d hr, %02d min, %02d sec'  %(get_hms(elapsed_time)))
        if scheduler is not None:
            scheduler.step()

criterion=nn.CrossEntropyLoss()
best_model=create_model(model_name = args.model, num_classes=num_classes).to(device)
if args.scan_para==0:
    resume_best= os.path.join(new_model_checkpoint_dir, f'{args.model_save_tag}.pth.tar')
else:
    resume_best= os.path.join(new_model_checkpoint_dir, f'{args.model_save_tag}_{args.entropy_percentile}_{args.alpha}.pth.tar')
assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model'
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best)
best_model.load_state_dict(checkpoint['state_dict'])

privateset = torch.utils.data.Subset(privateset, range(config.attack.num_sample))
testset = torch.utils.data.Subset(testset, range(config.attack.num_sample))
private_dataloader = torch.utils.data.DataLoader(privateset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
_,best_val = test(test_dataloader, best_model, criterion, use_cuda, device=device)
_,best_train = test(private_dataloader, best_model, criterion, use_cuda, device=device)
print('\t===>  %s  HAMP model: train acc %.4f val acc %.4f'%(resume_best, best_train, best_val), flush=True)
