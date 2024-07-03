import os
import random
import pickle
import numpy as np
import yaml
import torchvision



def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def main():
    DATASET_PATH = './data'
    dataset= torchvision.datasets.CIFAR10(DATASET_PATH, train=True, download=True)
    X=dataset.data.transpose(0,3,1,2)
    Y=np.array(dataset.targets)

    r=pickle.load(open('./cifar_shuffle.pkl','rb'))

    X=X[r]
    Y=Y[r]

    np.random.seed(0)

    train_data = X
    train_label = Y

    train_data_tr_attack = train_data[:int(0.5*len(r))]
    train_label_tr_attack = train_label[:int(0.5*len(r))]

    train_data_te_attack = train_data[int(0.5*len(r)):]
    train_label_te_attack = train_label[int(0.5*len(r)):]

    dataset= torchvision.datasets.CIFAR10(DATASET_PATH, train=False)
    test_X=dataset.data.transpose(0,3,1,2)
    test_Y=np.array(dataset.targets)

    np.random.seed(2000)

    all_test_data = test_X
    all_test_label = test_Y

    r = np.arange(len(test_X))
    np.random.shuffle(r)
    all_test_data=all_test_data[r]
    all_test_label=all_test_label[r]


    print(train_label_tr_attack[:20])
    print(train_label_te_attack[:20])

    path2 = os.path.join(DATASET_PATH, 'partition')
    if not os.path.isdir(path2):
        mkdir_p(path2)

    np.save(os.path.join(DATASET_PATH, 'partition', 'tr_data.npy'), train_data_tr_attack)
    np.save(os.path.join(DATASET_PATH, 'partition', 'tr_label.npy'), train_label_tr_attack)
    np.save(os.path.join(DATASET_PATH, 'partition', 'te_data.npy'), train_data_te_attack)
    np.save(os.path.join(DATASET_PATH, 'partition', 'te_label.npy'), train_label_te_attack)
    np.save(os.path.join(DATASET_PATH, 'partition', 'train_data.npy'), train_data)
    np.save(os.path.join(DATASET_PATH, 'partition', 'train_label.npy'), train_label)
    np.save(os.path.join(DATASET_PATH, 'partition', 'all_test_data.npy'), all_test_data)
    np.save(os.path.join(DATASET_PATH, 'partition', 'all_test_label.npy'), all_test_label)


if __name__ == '__main__':
    main()
