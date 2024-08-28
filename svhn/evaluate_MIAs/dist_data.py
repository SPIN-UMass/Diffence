import argparse
import os
parser = argparse.ArgumentParser() 
parser.add_argument('--config', type=str)  
parser.add_argument('--world-size', type=int, required=True)  
parser.add_argument('--rank', type=int)   
parser.add_argument('--diff', type=int, default='0')
parser.add_argument('--N', type=int, default=0) 
parser.add_argument('--T', type=int, default=0) 
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
import math
from model_factory import *
os.chdir('../')

assert torch.cuda.is_available()
torch.manual_seed(20240814)

def parse_config(config_path=None):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        new_config = dict2namespace(config)
    return new_config
config = parse_config(args.config)

if not args.N == 0:
    config.purification.path_number = args.N
if not args.T == 0:
    config.purification.purify_step = args.T

if not args.mode == 0:
    config.attack.mode=args.mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resume_best= config.attack.path

trainset, testset, privateset,refset, trainset_origin, privateset_origin, num_classes = get_dataset()
private_data_len= len(privateset) 
attack_tr_len=private_data_len 
attack_te_len=0  
tr_frac=0.5 
te_frac=0.5 

X = []
Y = []
X_test = []
Y_test = []
for item in trainset: 
    X.append( item[0].numpy() )
    Y.append( item[1]  )
for item in testset:
    X_test.append( item[0].numpy() )
    Y_test.append( item[1]  )
X = np.asarray(X)
Y = np.asarray(Y)
X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)

private_data=X[:private_data_len]
private_label=Y[:private_data_len]

ref_data = X[private_data_len:]
ref_label = Y[private_data_len:]

test_data=X_test[:]
test_label=Y_test[:]


# get private data and label tensors required to train the unprotected model
private_data_tensor=torch.from_numpy(private_data).type(torch.FloatTensor)
private_label_tensor=torch.from_numpy(private_label).type(torch.LongTensor)

# get reference data and label tensors required to distil the knowledge into the protected model
ref_data_tensor=torch.from_numpy(ref_data).type(torch.FloatTensor)
ref_label_tensor=torch.from_numpy(ref_label).type(torch.LongTensor)

#test
test_data_tensor=torch.from_numpy(test_data).type(torch.FloatTensor)
test_label_tensor=torch.from_numpy(test_label).type(torch.LongTensor)

# get member data and label tensors required to train MIA model
mia_train_members_data_tensor=private_data_tensor[:int(tr_frac*private_data_len)]
mia_train_members_label_tensor=private_label_tensor[:int(tr_frac*private_data_len)]

# get member data and label tensors required to validate MIA model
mia_test_members_data_tensor=private_data_tensor[int(tr_frac*private_data_len):int((tr_frac+te_frac)*private_data_len)]
mia_test_members_label_tensor=private_label_tensor[int(tr_frac*private_data_len):int((tr_frac+te_frac)*private_data_len)]

# # get non-members from ref data
# # get non-member data and label tensors required to train 
# mia_train_nonmembers_data_tensor = ref_data_tensor[:int(tr_frac*private_data_len)]
# mia_train_nonmembers_label_tensor = ref_label_tensor[:int(tr_frac*private_data_len)]
# # get member data and label tensors required to test MIA model
# mia_test_nonmembers_data_tensor = ref_data_tensor[int((tr_frac)*private_data_len):int((tr_frac+te_frac)*private_data_len)]
# mia_test_nonmembers_label_tensor =ref_label_tensor[int((tr_frac)*private_data_len):int((tr_frac+te_frac)*private_data_len)]

# get non-members from test data
# get non-member data and label tensors required to train 
test_data_len=len(test_data_tensor)
mia_train_nonmembers_data_tensor = test_data_tensor[:int(tr_frac*test_data_len)]
mia_train_nonmembers_label_tensor = test_label_tensor[:int(tr_frac*test_data_len)]
# get member data and label tensors required to test MIA model
mia_test_nonmembers_data_tensor = test_data_tensor[int((tr_frac)*test_data_len):int((tr_frac+te_frac)*test_data_len)]
mia_test_nonmembers_label_tensor =test_label_tensor[int((tr_frac)*test_data_len):int((tr_frac+te_frac)*test_data_len)]

## Tensors required to validate and test the unprotected and protected models
# get test data and label tensors
te_data_tensor=torch.from_numpy(test_data).type(torch.FloatTensor)
te_label_tensor=torch.from_numpy(test_label).type(torch.LongTensor)
 
num_sample=config.attack.num_sample
rank = args.rank
world_size = args.world_size
if not args.diff == 0:
    data_path = f'./evaluate_MIAs/data/{config.attack.target_model}/num_sample{config.attack.num_sample}/{config.attack.save_tag}/diff{args.diff}/iter{config.purification.max_iter}_path{config.purification.path_number}_step{config.purification.purify_step}'
else:
    data_path = f'./evaluate_MIAs/data/{config.attack.target_model}/num_sample{config.attack.num_sample}/{config.attack.save_tag}/diff{args.diff}'
os.makedirs(data_path, exist_ok=True)

def get_data_for_rank(all_data):
    num_per_rank = math.ceil(num_sample/world_size)
    if len(all_data)<num_sample:
        num_per_rank = math.ceil(len(all_data)/world_size)
    return all_data[:num_sample][rank*num_per_rank: (rank+1)*num_per_rank]
 
private_data_tensor,mia_train_members_data_tensor,mia_test_members_data_tensor,\
       mia_train_nonmembers_data_tensor,mia_test_nonmembers_data_tensor,\
       ref_data_tensor,te_data_tensor = get_data_for_rank(private_data_tensor), get_data_for_rank(mia_train_members_data_tensor),\
                                                                get_data_for_rank(mia_test_members_data_tensor), get_data_for_rank(mia_train_nonmembers_data_tensor),\
                                                                      get_data_for_rank(mia_test_nonmembers_data_tensor),\
                                                                get_data_for_rank(ref_data_tensor),get_data_for_rank(te_data_tensor)

private_label_tensor,mia_train_members_label_tensor,mia_test_members_label_tensor,\
       mia_train_nonmembers_label_tensor,mia_test_nonmembers_label_tensor,\
       ref_label_tensor,te_label_tensor = get_data_for_rank(private_label_tensor), get_data_for_rank(mia_train_members_label_tensor),\
                                                                get_data_for_rank(mia_test_members_label_tensor), get_data_for_rank(mia_train_nonmembers_label_tensor),\
                                                                      get_data_for_rank(mia_test_nonmembers_label_tensor),\
                                                                get_data_for_rank(ref_label_tensor),get_data_for_rank(te_label_tensor)

best_model=create_model(model_name = config.attack.target_model, num_classes=num_classes)
best_model=best_model.cuda()
if config.attack.save_tag.startswith('dpsgd'):
    from opacus.dp_model_inspector import DPModelInspector
    from opacus.utils import module_modification
    from opacus import PrivacyEngine
    best_model = module_modification.convert_batchnorm_modules(best_model).to(device) 
criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
assert os.path.isfile(resume_best), 'Error: no checkpoint directory %s found for best model'%resume_best
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best, map_location='cuda')
best_model.load_state_dict(checkpoint['state_dict'])
#YF
if args.diff == 1:
    best_model =  ModelwDiff(best_model,  args)
elif args.diff ==2: #pretrained model
    best_model =  ModelwDiff_v2(best_model, args)

BATCH_SIZE=100
if hasattr(best_model, 'config'):
    BATCH_SIZE = best_model.config.purification.bsize


# for direct query attack
tr_members = mia_train_members_data_tensor.numpy()
tr_members_y = mia_train_members_label_tensor.numpy()
tr_non_members =  mia_train_nonmembers_data_tensor.numpy()
tr_non_members_y = mia_train_nonmembers_label_tensor.numpy()
tr_m_true = np.r_[ np.ones(tr_members.shape[0]), np.zeros(tr_non_members.shape[0]) ]

def softmax_by_row(logits, T = 1.0):
    mx = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp((logits - mx)/T)
    denominator = np.sum(exp, axis=-1, keepdims=True)
    return exp/denominator

def _model_predictions(model, x, y, batch_size=256):
    model.eval()

    len_t = len(x)//batch_size
    if len(x)%batch_size:
        len_t += 1

    first = True
    if args.diff==0:
        for ind in range(len_t):
            outputs = model.forward(x[ind*batch_size:(ind+1)*batch_size].cuda())
            outputs = outputs.data.cpu().numpy()
            if(first):
                outs = outputs
                first=False
            else:
                outs = np.r_[outs, outputs]
        return (outs, np.argmax(outs,1), np.array(y))
    else:
        for ind in tq(range(len_t)):
            outputs,predicted_labels,ground_labels = model.forward(x[ind*batch_size:(ind+1)*batch_size].cuda(), y[ind*batch_size:(ind+1)*batch_size])
            if(first):
                outs = outputs
                p_labels = predicted_labels
                g_labels = ground_labels
                first=False
            else:
                outs = np.r_[outs, outputs]
                p_labels =np.r_[p_labels, predicted_labels]
                g_labels =np.r_[g_labels, ground_labels]
        return (outs, p_labels,g_labels) #p_labels: The predicted labels of original images    g_labels: The ground truth labels of orginal images
  

if(config.attack.prepMemGuard):
    from scipy.special import softmax

    if args.diff!=0:
        scope = config.attack.scope
        memguard_logit_path=os.path.join(data_path, f'logits_for_memguard_{config.attack.mode}.npz')
    else:
        memguard_logit_path=os.path.join(data_path, f'logits_for_memguard_diff0.npz')
    
    # path = os.path.join(data_path, f'diff_{world_size}_{rank}.npz')
    data = np.load(memguard_logit_path)
    shadow_train_performance = (softmax(data['shadow_train_performance_logits'],1),data['shadow_train_performance_logits'])
    shadow_test_performance = (softmax(data['shadow_test_performance_logits'],1),data['shadow_test_performance_logits'])
    target_train_performance =(softmax(data['target_train_performance_logits'],1),data['target_train_performance_logits'])
    target_test_performance = (softmax(data['target_test_performance_logits'],1),data['target_test_performance_logits'])

    shadow_train_performance = (get_data_for_rank(shadow_train_performance[0]),get_data_for_rank(shadow_train_performance[1]))
    shadow_test_performance = (get_data_for_rank(shadow_test_performance[0]), get_data_for_rank(shadow_test_performance[1]))
    target_train_performance =(get_data_for_rank(target_train_performance[0]), get_data_for_rank(target_train_performance[1]))
    target_test_performance = (get_data_for_rank(target_test_performance[0]), get_data_for_rank(target_test_performance[1]))

    import numpy as np
    np.random.seed(1000)
    import imp
    import keras
    from keras.models import Model
    import tensorflow.compat.v1 as tf
    import os
    import configparser
    import argparse
    from scipy.special import softmax 
    tf.disable_eager_execution()

    user_label_dim=config.trainer.num_class
    num_classes=1

    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.3
    config_gpu.gpu_options.visible_device_list = "0"

    sess = tf.InteractiveSession(config=config_gpu)
    sess.run(tf.global_variables_initializer())
    train_outputs1 = shadow_train_performance[0] # score for first half of members for training attack (train_member_pred)
                                                                        # this is the known member set for adversary
    len1 = len(train_outputs1)
    train_outputs2 = target_train_performance[0]  # score for the second half of members for evaluating attack (test_member_pred)
                                                    # this is the actual members for evaluating the attack  
    len2 = len(train_outputs2)
    train_outputs = np.concatenate((train_outputs1, train_outputs2))
    test_outputs1 = shadow_test_performance[0] # data that were used for training the attack (these are not members)
                                                    # this is set to 0.1 (members) vs. 0.15 (ref data)
                                                    # this is the known non-member set for the adversary   
    test_outputs2 = target_test_performance[0] # test set 
                                                # this is the actual non-members for evaluating the attack
    test_outputs = np.concatenate((test_outputs1, test_outputs2))
    train_logits1 = shadow_train_performance[1]
    train_logits2 = target_train_performance[1]
    train_logits = np.concatenate((train_logits1, train_logits2))
    test_logits1 = shadow_test_performance[1]
    test_logits2 = target_test_performance[1]
    test_logits = np.concatenate((test_logits1, test_logits2))
    min_len = min(len(train_outputs), len(test_outputs))
    print('selected number of members and non-members are: ', min(len(train_outputs), len(test_outputs)), min(len(train_logits), len(test_logits)))

    f_evaluate = np.concatenate((train_outputs[:min_len], test_outputs[:min_len]))
    f_evaluate_logits = np.concatenate((train_logits[:min_len], test_logits[:min_len]))
    l_evaluate = np.zeros(len(f_evaluate))
    l_evaluate[:min_len] = 1
    print('dataset shape information: ', f_evaluate.shape, f_evaluate_logits.shape, l_evaluate.shape, min_len)

    f_evaluate_origin=np.copy(f_evaluate)  #keep a copy of original one
    f_evaluate_logits_origin=np.copy(f_evaluate_logits)
    #############as we sort the prediction sscores, back_index is used to get back original scores#############
    sort_index=np.argsort(f_evaluate,axis=1)
    back_index=np.copy(sort_index)
    for i in np.arange(back_index.shape[0]):
        back_index[i,sort_index[i,:]]=np.arange(back_index.shape[1])
    f_evaluate=np.sort(f_evaluate,axis=1)
    f_evaluate_logits=np.sort(f_evaluate_logits,axis=1)


    print("f evaluate shape: {}".format(f_evaluate.shape))
    print("f evaluate logits shape: {}".format(f_evaluate_logits.shape))


    ##########loading defense model -------------------------------------------------------------
    import tensorflow.keras as keras
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation, Input, concatenate
    import numpy as np
    import tensorflow as tf
    def model_defense_optimize(input_shape,labels_dim):
        inputs_b=Input(shape=input_shape)
        x_b=Activation('softmax')(inputs_b)
        x_b=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
        x_b=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
        x_b=Dense(64,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
        outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
        outputs=Activation('sigmoid')(outputs_pre)
        model = Model(inputs=inputs_b, outputs=outputs)
        return model

    from tensorflow.keras.models import load_model
    defense_model_path = f'./evaluate_MIAs/memguard/{config.attack.target_model}'
    # defense_model_path = os.path.join(defense_model_path, f'memguard_diff{args.diff}_{config.attack.save_tag}')
    defense_model_path = os.path.join(defense_model_path, f'memguard_diff0_{config.attack.save_tag}')
    defense_model = load_model( defense_model_path )
    weights=defense_model.get_weights()
    del defense_model

    input_shape=f_evaluate.shape[1:]
    print("Loading defense model...")

    model=model_defense_optimize(input_shape=input_shape,labels_dim=num_classes)
    model.compile(loss=keras.losses.binary_crossentropy,optimizer=tf.keras.optimizers.SGD(lr=0.001),metrics=['accuracy'])
    model.set_weights(weights)
    # model.load_weights(defense_model_path)
    model.trainable=False

    memguard_start=time.time()
    import tensorflow.compat.v1 as tf
    ########evaluate the performance of defense's attack model on undefended data########
    scores_evaluate = model.evaluate(f_evaluate_logits, l_evaluate, verbose=0)
    print('\nevaluate loss on model:', scores_evaluate[0])
    print('==>\tevaluate the NN attack accuracy on model (undefended data):', scores_evaluate[1],flush=True)   # means MemGuard's attack model's attack accuracy on the undefended data

    output=model.layers[-2].output[:,0]
    c1=1.0  #used to find adversarial examples 
    c2=10.0    #penalty such that the index of max score is keeped
    c3=0.1
    #alpha_value=0.0 

    origin_value_placeholder=tf.placeholder(tf.float32,shape=(1,user_label_dim)) #placeholder with original confidence score values (not logit)
    label_mask=tf.placeholder(tf.float32,shape=(1,user_label_dim))  # one-hot encode that encodes the predicted label 
    c1_placeholder=tf.placeholder(tf.float32)
    c2_placeholder=tf.placeholder(tf.float32)
    c3_placeholder=tf.placeholder(tf.float32)

    correct_label = tf.reduce_sum(label_mask * model.input, axis=1)
    wrong_label = tf.reduce_max((1-label_mask) * model.input - 1e8*label_mask, axis=1)


    loss1=tf.abs(output)
    ### output of defense classifier is the logit, when it is close to 0, the prediction by the inference is close to 0.5, i.e., random guess.
    ### loss1 ensures random guessing for inference classifier ###
    loss2=tf.nn.relu(wrong_label-correct_label)
    ### loss2 ensures no changes to target classifier predictions ###
    loss3=tf.reduce_sum(tf.abs(tf.nn.softmax(model.input)-origin_value_placeholder)) #L-1 norm
    ### loss3 ensures minimal noise addition

    loss=c1_placeholder*loss1+c2_placeholder*loss2+c3_placeholder*loss3
    gradient_targetlabel=K.gradients(loss,model.input)
    label_mask_array=np.zeros([1,user_label_dim],dtype=np.float)
    ##########################################################
    result_array=np.zeros(f_evaluate.shape,dtype=np.float)
    result_array_logits=np.zeros(f_evaluate.shape,dtype=np.float)
    success_fraction=0.0
    max_iteration=300   #max iteration if can't find adversarial example that satisfies requirements
    np.random.seed(1000)
    for test_sample_id in np.arange(0,f_evaluate.shape[0]):
        if test_sample_id%100==0:
            print("test sample id: {}".format(test_sample_id),flush=True)
        max_label=np.argmax(f_evaluate[test_sample_id,:])
        origin_value=np.copy(f_evaluate[test_sample_id,:]).reshape(1,user_label_dim)
        origin_value_logits=np.copy(f_evaluate_logits[test_sample_id,:]).reshape(1,user_label_dim)
        label_mask_array[0,:]=0.0
        label_mask_array[0,max_label]=1.0
        sample_f=np.copy(origin_value_logits)
        result_predict_scores_initial=model.predict(sample_f)
        ########## if the output score is already very close to 0.5, we can just use it for numerical reason
        if np.abs(result_predict_scores_initial-0.5)<=1e-5:
            success_fraction+=1.0
            result_array[test_sample_id,:]=origin_value[0,back_index[test_sample_id,:]]
            result_array_logits[test_sample_id,:]=origin_value_logits[0,back_index[test_sample_id,:]]
            continue
        last_iteration_result=np.copy(origin_value)[0,back_index[test_sample_id,:]]
        last_iteration_result_logits=np.copy(origin_value_logits)[0,back_index[test_sample_id,:]]
        success=True
        c3=0.1
        iterate_time=1
        while success==True: 
            sample_f=np.copy(origin_value_logits)
            j=1
            result_max_label=-1
            result_predict_scores=result_predict_scores_initial
            while j<max_iteration and (max_label!=result_max_label or (result_predict_scores-0.5)*(result_predict_scores_initial-0.5)>0):
                gradient_values=sess.run(gradient_targetlabel,feed_dict={model.input:sample_f,origin_value_placeholder:origin_value,label_mask:label_mask_array,c3_placeholder:c3,c1_placeholder:c1,c2_placeholder:c2})[0][0]
                gradient_values=gradient_values/np.linalg.norm(gradient_values)
                sample_f=sample_f-0.1*gradient_values
                result_predict_scores=model.predict(sample_f)
                result_max_label=np.argmax(sample_f)
                j+=1        
            if max_label!=result_max_label:
                if iterate_time==1:
                    print("failed sample for label not same for id: {},c3:{} not add noise".format(test_sample_id,c3))
                    success_fraction-=1.0
                break                
            if ((model.predict(sample_f)-0.5)*(result_predict_scores_initial-0.5))>0:
                if iterate_time==1:
                    print("max iteration reached with id: {}, max score: {}, prediction_score: {}, c3: {}, not add noise".format(test_sample_id,np.amax(softmax(sample_f)),result_predict_scores,c3))
                break
            last_iteration_result[:]=softmax(sample_f)[0,back_index[test_sample_id,:]]
            last_iteration_result_logits[:]=sample_f[0,back_index[test_sample_id,:]]
            iterate_time+=1 
            c3=c3*10
            if c3>100000:
                break
        success_fraction+=1.0
        result_array[test_sample_id,:]=last_iteration_result[:]
        result_array_logits[test_sample_id,:]=last_iteration_result_logits[:]
    print("Success fraction: {}".format(success_fraction/float(f_evaluate.shape[0])))
    memguard_end=time.time()
    print(f"Memguard time: {memguard_end-memguard_start}")

    scores_evaluate = model.evaluate(result_array_logits, l_evaluate, verbose=0)
    print('evaluate loss on model:', scores_evaluate[0])
    print('\n====> evaluate accuracy on model:', scores_evaluate[1])

    if args.diff!=0:
        scope = config.attack.scope
        file_path=os.path.join(data_path, 'memguard_defense_results',f'memguard_{config.attack.mode}')
    else:
        file_path=os.path.join(data_path, 'memguard_defense_results',f'memguard_diff0')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        

    np.savez(os.path.join(file_path, 'purchase_shadow_defense.npz'), defense_output=result_array, defense_logits = result_array_logits, 
                tc_outputs=f_evaluate_origin)


    # these the conf scores after memguard
    np.save(os.path.join(file_path, f'memguard_known_member_{world_size}_{rank}.npy'), result_array[:len1])
    np.save(os.path.join(file_path, f'memguard_test_member_{world_size}_{rank}.npy'), result_array[len1:len1+len2])
    np.save(os.path.join(file_path, f'memguard_known_nonmember_{world_size}_{rank}.npy'), result_array[len1+len2:len1+len2+len1])
    np.save(os.path.join(file_path, f'memguard_test_non_member_{world_size}_{rank}.npy'), result_array[len1+len2+len1:])

    np.save(os.path.join(file_path, f'memguard_known_member_logit_{world_size}_{rank}.npy'), result_array_logits[:len1])
    np.save(os.path.join(file_path, f'memguard_test_member_logit_{world_size}_{rank}.npy'), result_array_logits[len1:len1+len2])
    np.save(os.path.join(file_path, f'memguard_known_nonmember_logit_{world_size}_{rank}.npy'), result_array_logits[len1+len2:len1+len2+len1])
    np.save(os.path.join(file_path, f'memguard_test_non_member_logit_{world_size}_{rank}.npy'), result_array_logits[len1+len2+len1:])
    sys.exit() 
# for test target model 
else:
    test_target_train_performance = _model_predictions(best_model, private_data_tensor,private_label_tensor, batch_size=BATCH_SIZE)
    test_target_test_performance = _model_predictions(best_model,te_data_tensor,te_label_tensor, batch_size=BATCH_SIZE)

    # for attack data
    shadow_model = best_model
    shadow_train_performance = _model_predictions(shadow_model, torch.from_numpy(tr_members).type(torch.FloatTensor), torch.from_numpy(tr_members_y).type(torch.LongTensor),
                                                         batch_size=BATCH_SIZE)
    shadow_test_performance = _model_predictions(shadow_model, torch.from_numpy(tr_non_members).type(torch.FloatTensor), torch.from_numpy(tr_non_members_y).type(torch.LongTensor),
                                                     batch_size=BATCH_SIZE)      

    target_train_performance = _model_predictions(best_model, mia_test_members_data_tensor, mia_test_members_label_tensor,  batch_size=BATCH_SIZE)
    target_test_performance = _model_predictions(best_model, mia_test_nonmembers_data_tensor, mia_test_nonmembers_label_tensor,   batch_size=BATCH_SIZE)
    
np.savez(os.path.join(data_path, f'diff_{world_size}_{rank}.npz'),\
                shadow_train_performance_logits=shadow_train_performance[0], shadow_train_performance_plabels = shadow_train_performance[1],shadow_train_performance_glabels = shadow_train_performance[2],\
                        shadow_test_performance_logits=shadow_test_performance[0],shadow_test_performance_plabels=shadow_test_performance[1], shadow_test_performance_glabels=shadow_test_performance[2],\
                        target_train_performance_logits=target_train_performance[0],target_train_performance_plabels=target_train_performance[1], target_train_performance_glabels=target_train_performance[2],\
                            target_test_performance_logits=target_test_performance[0],target_test_performance_plabels=target_test_performance[1], target_test_performance_glabels=target_test_performance[2],\
                                test_target_train_performance_logits=test_target_train_performance[0],test_target_train_performance_plabels=test_target_train_performance[1], test_target_train_performance_glabels=test_target_train_performance[2],\
                                test_target_test_performance_logits=test_target_test_performance[0],test_target_test_performance_plabels=test_target_test_performance[1], test_target_test_performance_glabels=test_target_test_performance[2])


