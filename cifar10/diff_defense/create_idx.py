import pickle
import numpy as np
import os

# DM and classifier trained on the same half
if not os.path.isfile('./cifar_shuffle.pkl'):
    splits = np.load(f'./diff_ckpt/CIFAR10_train_ratio0.5.npz')
    nonmember_idxs = splits['mia_eval_idxs']
    member_idxs = splits['mia_train_idxs']
    np.random.shuffle(member_idxs)
    np.random.shuffle(nonmember_idxs)
    all_indices = np.concatenate((member_idxs,nonmember_idxs))
    pickle.dump(all_indices,open('./cifar_shuffle.pkl','wb'))
else:
    all_indices=pickle.load(open('./cifar_shuffle.pkl','rb'))

# # DM and classifier trained on the different half
# if not os.path.isfile('./cifar_shuffle.pkl'):
#     splits = np.load(f'./diff_ckpt/CIFAR10_train_ratio0.5.npz')
#     nonmember_idxs = splits['mia_eval_idxs']
#     member_idxs = splits['mia_train_idxs']
#     remaining_idxs = np.arange(50000,60000)
#     np.random.shuffle(member_idxs)
#     np.random.shuffle(nonmember_idxs)
#     np.random.shuffle(remaining_idxs)
#     all_indices = np.concatenate((nonmember_idxs,member_idxs,remaining_idxs))
#     pickle.dump(all_indices,open('./cifar_shuffle.pkl','wb'))
# else:
#     all_indices=pickle.load(open('./cifar_shuffle.pkl','rb'))
