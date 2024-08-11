import sys
import os 
import argparse
parser = argparse.ArgumentParser() 
parser.add_argument('--train_size', type=int, default=25000)
parser.add_argument('--train_org', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--model_save_tag', type=str, default='dpsgd', help='a tag to be appended to the saved model path')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--max_norm', type=float, default=1.2, help='grad norm clipping bound')
parser.add_argument('--sigma', type=float, default=0.01, help='noise multiplier (small value for better utility)')
parser.add_argument('--config', type=str, default='./configs/default8.yml')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
sys.path.insert(0,'./util/')
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
from torch.distributions import Categorical
from model_factory import *

from opacus.dp_model_inspector import DPModelInspector
from opacus.utils import module_modification
from opacus import PrivacyEngine


config = parse_config(args.config)
args.model = config.trainer.classifier
org_model_checkpoint_dir=f'./final-all-models/{config.trainer.classifier}'

float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
user_lr=args.lr
if(not os.path.exists(org_model_checkpoint_dir)):
    os.makedirs(org_model_checkpoint_dir,exist_ok=True)
is_train_org = args.train_org
BATCH_SIZE =config.trainer.batch_size
private_data_len= int(args.train_size) 
ref_data_len = int(args.train_size) 
num_epochs=config.trainer.classifier_epochs

args.max_norm = config.trainer.max_norm
args.sigma = config.trainer.sigma


import random
random.seed(1)
# prepare test data parts
trainset, testset, privateset,refset,trainset_origin, privateset_origin,num_classes = get_dataset(aug=config.trainer.augmentation)

print("loading data")
def adjust_learning_rate(optimizer, epoch):
    global state
    # if epoch in [90, 120, 160]: 
    if epoch in [30,60,90]: 
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.

private_dataloader = torch.utils.data.DataLoader(privateset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

print('tr len %d | val data len %d'%
      (len(privateset),len(testset)), flush=True)


def train(train_loader,model,criterion,optimizer,epoch,use_cuda,num_batchs=999999,batch_size=32, uniform_reg=False):
    # switch to train mode
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for ind, (inputs,targets) in enumerate(train_loader): 
        inputs, targets = inputs.to(device), targets.to(device)
        outputs  = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg, top1.avg)


criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
model=create_model(model_name = args.model, num_classes=num_classes)

# optimizer = optim.SGD(model.parameters(), lr=user_lr, momentum=0.9, weight_decay=1e-4)
# optimizer = optim.Adam(model.parameters(), lr = 0.001, betas = (0.9, 0.999), amsgrad = True, weight_decay=1e-6)
optimizer, scheduler = parser_opt_scheduler(model,config)


model = module_modification.convert_batchnorm_modules(model)
inspector = DPModelInspector()
assert inspector.validate(model)
model=model.to(device)

privacy_engine = PrivacyEngine(
        model,
        batch_size=BATCH_SIZE,
        sample_size=len(privateset),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=args.sigma,
        max_grad_norm=args.max_norm,
    )
privacy_engine.attach(optimizer)


print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
best_val_acc=0

import time
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s
elapsed_time = 0
if(is_train_org): 
    
    for epoch in range(num_epochs):
        start_time = time.time()
        # adjust_learning_rate(optimizer, epoch) 
        train_loss, train_acc = train(private_dataloader, model, criterion, optimizer, epoch, use_cuda, batch_size= BATCH_SIZE, uniform_reg=False)
        val_loss, val_acc = test(test_dataloader, model, criterion, use_cuda, device=device)

        is_best = val_acc > best_val_acc
        best_val_acc=max(val_acc, best_val_acc)
    
        # save_checkpoint_global(
        #     {
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'best_acc': best_val_acc,
        #         'optimizer': optimizer.state_dict(),
        #     },
        #     is_best,
        #     checkpoint=org_model_checkpoint_dir,
        #     filename=f"{args.model_save_tag}-trainSize-{args.train_size}-epoch{epoch}.pth.tar",
        #     best_filename=f'{args.model_save_tag}-trainSize-{args.train_size}.pth.tar',
        # )
        #print(optimizer.param_groups[0]['lr'])
        print('  epoch %d | tr acc %.2f loss %.2f | val acc %.2f loss %.2f |  best val acc %.2f '
              %(epoch, train_acc, train_loss, val_acc, val_loss, best_val_acc,), flush=True)

        print()
        
        elapsed_time += time.time() - start_time
        print('| Elapsed time : %d hr, %02d min, %02d sec'  %(get_hms(elapsed_time)))

        if scheduler is not None:
            scheduler.step()


criterion=nn.CrossEntropyLoss()
best_model=create_model(model_name = args.model, num_classes=num_classes)
best_model = module_modification.convert_batchnorm_modules(best_model).to(device)
resume_best=os.path.join(org_model_checkpoint_dir,f'{args.model_save_tag}-trainSize-{args.train_size}.pth.tar')
assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model'
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best)
best_model.load_state_dict(checkpoint['state_dict'])
_,best_val = test(test_dataloader, best_model, criterion, use_cuda, device=device)
_,best_train = test(private_dataloader, best_model, criterion, use_cuda, device=device)
print('\t===> Undefended model %s | train acc %.4f | val acc %.4f '%(resume_best, best_train, best_val), flush=True)


