# from https://github.com/cchoquette/membership-inference/blob/main/attack_features.py

import numpy as np
import scipy.ndimage.interpolation as interpolation
# import tensorflow as tf
from .aug_utils import apply_augment, create_rotates, create_translates, softmax, get_data
import torch
import torch.nn.functional as F
from PIL import Image

class Cifardata(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.transform = transform
        self.labels = labels

    def __getitem__(self, index):
        img =  Image.fromarray(((255*self.data[index]).transpose(1,2,0).astype(np.uint8)))
        label = self.labels[index]
        img = self.transform(img)

        return img, label   

    def __len__(self):
        return len(self.labels)

def check_correct(ds, predictions):
  """Used for augmentation MI attack to check if each image was correctly classified using label-only access.

  Args:
    ds: tuple of (images, labels) where images are (N, H, W, C).
    predictions: predictions from model.

  Returns: 1 if correct, 0 if incorrect for each sample.

  """
  return np.equal(ds[1].flatten(), np.argmax(predictions, axis=1)).squeeze()


def augmentation_attack(config, model, train_set, test_set, batch=100, transform_test=None):

  attack_type = config.label_only.aug_attack_type
  augment_kwarg = config.label_only.aug_augment_kwarg

  max_samples = len(train_set[0])
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if attack_type == 'r':
    augments = create_rotates(augment_kwarg)
  elif attack_type == 'd':
    augments = create_translates(augment_kwarg)
  else:
    raise ValueError(f"attack type_: {attack_type} is not valid.")
  m = np.concatenate([np.ones(max_samples),
                      np.zeros(max_samples)], axis=0)
  attack_in = np.zeros((max_samples, len(augments)))
  attack_out = np.zeros((max_samples, len(augments)))
  for i, augment in enumerate(augments):
    train_augment = apply_augment(train_set, augment, attack_type)
    test_augment = apply_augment(test_set, augment, attack_type)
    # train_ds = tf.data.Dataset.from_tensor_slices(train_augment).batch(batch)
    # test_ds = tf.data.Dataset.from_tensor_slices(test_augment).batch(batch)

    train_ds = Cifardata(train_augment[0], train_augment[1], transform_test)
    test_ds = Cifardata(test_augment[0], test_augment[1], transform_test)        

    t_knowtrainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=False)
    t_unknowtrainloader = torch.utils.data.DataLoader(test_ds, batch_size=batch, shuffle=False)
    
    train_out=[]
    test_out=[]
    for batch_ind, (inputs, targets) in enumerate(t_knowtrainloader):
      inputs = inputs.to(device, torch.float)
      targets = targets.to(device, torch.long)

      outputs = model(inputs)
      train_out.extend((F.softmax(outputs,dim=1)).detach().cpu().numpy())

    for batch_ind, (inputs, targets) in enumerate(t_unknowtrainloader):
      inputs = inputs.to(device, torch.float)
      targets = targets.to(device, torch.long)

      outputs = model(inputs)
      test_out.extend((F.softmax(outputs,dim=1)).detach().cpu().numpy())
    in_= np.array(train_out)
    out_ = np.array(test_out)
    # in_ = softmax(model.predict(train_ds))
    # out_ = softmax(model.predict(test_ds))
    attack_in[:, i] = check_correct(train_set, in_)
    attack_out[:, i] = check_correct(test_set, out_)
  attack_set = (np.concatenate([attack_in, attack_out], 0),
                np.concatenate([train_set[1], test_set[1]], 0),
                m)
  
  return attack_set


# target_train_set, target_test_set, source_train_set, source_test_set, input_dim, n_classes = get_data('cifar10', 1000)
# augmentation_attack(None, target_train_set, target_test_set, 1000,
#                                                             'd',
#                                                        9, 100)
