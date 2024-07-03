import torch.nn.functional as F
import torch
import numpy as np

def gen_ll(x_ll, network_clf, transform_raw_to_clf, config):
    # z: list of lists that consist of logits
    logit_ll = []
    softmax_ll = []
    onehot_ll = []
    for i in range(len(x_ll)):
        chain_length = len(x_ll[i])
        logit_l = []
        softmax_l = []
        onehot_l = []
        if config.classification.classify_all_steps:
            for j in range(chain_length):
                x_t = x_ll[i][j].clone().detach()
                logit_l.append(network_clf(transform_raw_to_clf(x_t)).detach().to('cpu'))
                # logit_l.append(network_clf(x_t))
                softmax_l.append(F.softmax(logit_l[j], dim=1))
                onehot_l.append(torch.argmax(softmax_l[j], dim=1))
        else:
            x_t = x_ll[i][-1].detach()
            logit_l.append(network_clf(transform_raw_to_clf(x_t)).detach().to('cpu'))
            # logit_l.append(network_clf(x_t))
            softmax_l.append(F.softmax(logit_l[0], dim=1))
            onehot_l.append(torch.argmax(softmax_l[0], dim=1))
        logit_ll.append(logit_l)
        softmax_ll.append(softmax_l)
        onehot_ll.append(onehot_l)
    return {"logit": logit_ll, "softmax": softmax_ll, "onehot": onehot_ll}

def acc_all_step(truth_ll, ground_label, config):
    # accuracy, involving all step
    total_logit = torch.zeros_like(truth_ll["logit"][0][0])
    total_softmax = torch.zeros_like(truth_ll["softmax"][0][0])
    total_onehot = torch.zeros_like(truth_ll["softmax"][0][0]) # NOT typo, "softmax" is 
    # asmpytotics of accuracy by increasing noise injected samples
    list_noisy_inputs_logit = [] # [1 noisy input, 2 noisy inputs, 3 noisy inputs, ...]
    list_noisy_inputs_softmax = []
    list_noisy_inputs_onehot = []
    for i in range(len(truth_ll["logit"])): # list of list of [bsize, nClass]
        for j in range(len(truth_ll["logit"][i])):
            total_logit += truth_ll["logit"][i][j] / len(truth_ll["logit"][i])
        list_noisy_inputs_logit.append(torch.eq(torch.argmax(total_logit, dim=1), ground_label).sum().float().to('cpu').numpy())
    for i in range(len(truth_ll["softmax"])): # list of list of [bsize, nClass]
        for j in range(len(truth_ll["softmax"][i])):
            total_softmax += truth_ll["softmax"][i][j] / len(truth_ll["softmax"][i])
        list_noisy_inputs_softmax.append(torch.eq(torch.argmax(total_softmax, dim=1), ground_label).sum().float().to('cpu').numpy())
    for i in range(len(truth_ll["onehot"])): # list of list of [bsize]
        for j in range(len(truth_ll["onehot"][i])):
            for k in range(truth_ll["onehot"][i][j].shape[0]):
                total_onehot[k, truth_ll["onehot"][i][j][k]] += 1./len(truth_ll["onehot"][i])
        list_noisy_inputs_onehot.append(torch.eq(torch.argmax(total_onehot, dim=1), ground_label).sum().float().to('cpu').numpy())
    max_purens = config.purification.max_iter
    # asmpytotics of accuracy by noise ensemble
    list_pur_steps_logit = [] # [first 1 steps, [1:2] steps, [1:3] steps, ...]
    list_pur_steps_softmax = []
    list_pur_steps_onehot = []
    list_each_step_logit = [] # [first 1 step, 2 step, 3 step, ...]
    list_each_step_softmax = []
    list_each_step_onehot = []
    total_logit_last_step = torch.zeros_like(total_logit) # [Last step]
    total_softmax_last_step = torch.zeros_like(total_softmax)
    total_onehot_last_step = torch.zeros_like(total_softmax)
    for j in range(max_purens):
        total_logit_each_step = torch.zeros_like(total_logit)
        total_softmax_each_step = torch.zeros_like(total_softmax)
        total_onehot_each_step = torch.zeros_like(total_onehot)
        for i in range(len(truth_ll["logit"])):
            total_logit_last_step += truth_ll["logit"][i][min(j, len(truth_ll["logit"][i])-1)]
            total_logit_each_step += truth_ll["logit"][i][min(j, len(truth_ll["logit"][i])-1)]
        list_pur_steps_logit.append(torch.eq(torch.argmax(total_logit_last_step, dim=1), ground_label).sum().float().to('cpu').numpy())
        list_each_step_logit.append(torch.eq(torch.argmax(total_logit_each_step, dim=1), ground_label).sum().float().to('cpu').numpy())
        for i in range(len(truth_ll["softmax"])):
            total_softmax_last_step += truth_ll["softmax"][i][min(j, len(truth_ll["softmax"][i])-1)]
            total_softmax_each_step += truth_ll["softmax"][i][min(j, len(truth_ll["softmax"][i])-1)]
        list_pur_steps_softmax.append(torch.eq(torch.argmax(total_softmax_last_step, dim=1), ground_label).sum().float().to('cpu').numpy())
        list_each_step_softmax.append(torch.eq(torch.argmax(total_softmax_each_step, dim=1), ground_label).sum().float().to('cpu').numpy())
        for i in range(len(truth_ll["onehot"])):
            for k in range(truth_ll["onehot"][i][min(j, len(truth_ll["onehot"][i])-1)].shape[0]):
                total_onehot_last_step[k, truth_ll["onehot"][i][min(j, len(truth_ll["onehot"][i])-1)][k]] += 1.
                total_onehot_each_step[k, truth_ll["onehot"][i][min(j, len(truth_ll["onehot"][i])-1)][k]] += 1.
        list_pur_steps_onehot.append(torch.eq(torch.argmax(total_onehot_last_step, dim=1), ground_label).sum().float().to('cpu').numpy())
        list_each_step_onehot.append(torch.eq(torch.argmax(total_onehot_each_step, dim=1), ground_label).sum().float().to('cpu').numpy())
    logit_correct = torch.eq(torch.argmax(total_logit, dim=1), ground_label).sum().float().to('cpu').numpy()
    softmax_correct = torch.eq(torch.argmax(total_softmax, dim=1), ground_label).sum().float().to('cpu').numpy()
    onehot_correct = torch.eq(torch.argmax(total_onehot, dim=1), ground_label).sum().float().to('cpu').numpy()
    list_noisy_inputs_logit = np.array(list_noisy_inputs_logit)
    list_noisy_inputs_softmax = np.array(list_noisy_inputs_softmax)
    list_noisy_inputs_onehot = np.array(list_noisy_inputs_onehot)
    list_pur_steps_logit = np.array(list_pur_steps_logit)
    list_pur_steps_softmax = np.array(list_pur_steps_softmax)
    list_pur_steps_onehot = np.array(list_pur_steps_onehot)

    return {"logit": logit_correct, "softmax": softmax_correct, "onehot": onehot_correct}, \
           {"logit": list_noisy_inputs_logit, "softmax": list_noisy_inputs_softmax, "onehot": list_noisy_inputs_onehot}, \
           {"logit": list_pur_steps_logit, "softmax": list_pur_steps_softmax, "onehot": list_pur_steps_onehot}, \
           {"logit": list_each_step_logit, "softmax": list_each_step_softmax, "onehot": list_each_step_onehot}

def acc_final_step(truth_ll, ground_label):
    # output: final_correct, correct
    total_logit = torch.zeros_like(truth_ll["logit"][0][0])
    total_softmax = torch.zeros_like(truth_ll["softmax"][0][0])
    total_onehot = torch.zeros_like(truth_ll["softmax"][0][0]) # NOT typo, "softmax" is right
    list_noisy_inputs_logit = []
    list_noisy_inputs_softmax = []
    list_noisy_inputs_onehot = []
    # accuracy, involving final step
    for i in range(len(truth_ll["logit"])): # list of list of [bsize, nClass]
        total_logit += truth_ll["logit"][i][-1]
        list_noisy_inputs_logit.append(torch.eq(torch.argmax(total_logit, dim=1), ground_label).sum().float().to('cpu').numpy())
    for i in range(len(truth_ll["softmax"])): # list of list of [bsize, nClass]
        total_softmax += truth_ll["softmax"][i][-1]
        list_noisy_inputs_softmax.append(torch.eq(torch.argmax(total_softmax, dim=1), ground_label).sum().float().to('cpu').numpy())
    for i in range(len(truth_ll["onehot"])): # list of list of [bsize]
        for k in range(truth_ll["onehot"][i][0].shape[0]):
            total_onehot[k][truth_ll["onehot"][i][-1][k]] += 1.
        list_noisy_inputs_onehot.append(torch.eq(torch.argmax(total_onehot, dim=1), ground_label).sum().float().to('cpu').numpy())
    logit_correct = torch.eq(torch.argmax(total_logit, dim=1), ground_label).sum().float().to('cpu').numpy()
    softmax_correct = torch.eq(torch.argmax(total_softmax, dim=1), ground_label).sum().float().to('cpu').numpy()
    onehot_correct = torch.eq(torch.argmax(total_onehot, dim=1), ground_label).sum().float().to('cpu').numpy()
    list_noisy_inputs_logit = np.array(list_noisy_inputs_logit)
    list_noisy_inputs_softmax = np.array(list_noisy_inputs_softmax)
    list_noisy_inputs_onehot = np.array(list_noisy_inputs_onehot)
    logit_result = torch.argmax(total_logit, dim=1).to('cpu').numpy()
    return {"logit": logit_correct, "softmax": softmax_correct, "onehot": onehot_correct}, \
           {"logit": list_noisy_inputs_logit, "softmax": list_noisy_inputs_softmax, "onehot": list_noisy_inputs_onehot}, logit_result

def output_final_step(truth_ll, ground_label, metric='min'):
    # output: final_correct, correct
    total_logit = torch.zeros_like(truth_ll["logit"][0][0])
    list_noisy_inputs_logit = []

    # accuracy, involving final step
    num=0
    logit_all_path=[]
    for i in range(len(truth_ll["logit"])): # list of list of [bsize, nClass]
        # total_logit += truth_ll["logit"][i][-1]
        logit_all_path.append(truth_ll["logit"][i][-1].detach().to('cpu').numpy())
        list_noisy_inputs_logit.append(torch.eq(torch.argmax(total_logit, dim=1), ground_label).sum().float().to('cpu').numpy())
        num+=1
    # logit_correct = torch.eq(torch.argmax(total_logit, dim=1), ground_label).sum().float().to('cpu').numpy()
    # list_noisy_inputs_logit = np.array(list_noisy_inputs_logit)
    # output = (total_logit/num).detach().to('cpu').numpy()
    logit_all_path = np.array(logit_all_path)
    if metric=='mean':
        output = np.mean(logit_all_path,axis=0)
    elif metric == 'max':
        output = np.max(logit_all_path,axis=0)
    elif metric == 'min':
        output = np.min(logit_all_path,axis=0)
    elif metric == 'median':
        output = np.median(logit_all_path,axis=0)
    return output

def output_final_step_tensor(truth_ll, metric='mean'):
    # output: final_correct, correct
    # accuracy, involving final step
    num=0
    logit_all_path=[]
    for i in range(len(truth_ll["logit"])): # list of list of [bsize, nClass]
        # total_logit += truth_ll["logit"][i][-1]
        logit_all_path.append(truth_ll["logit"][i][-1])
        num+=1
    # logit_correct = torch.eq(torch.argmax(total_logit, dim=1), ground_label).sum().float().to('cpu').numpy()
    # list_noisy_inputs_logit = np.array(list_noisy_inputs_logit)
    # output = (total_logit/num).detach().to('cpu').numpy()
    logit_all_path = torch.stack(logit_all_path)

    if metric=='mean':
        output = torch.mean(logit_all_path,axis=0)
    elif metric == 'max':
        logit_all_path1=logit_all_path.transpose(0,1)
        standard=torch.max(logit_all_path1,axis=2)[0]
        idx=torch.argmax(standard,1)
        logit_all_path_select = [logit_all_path1[i][idx[i]] for i in range(len(idx))]
        output = torch.stack(logit_all_path_select)
        # logit_all_path=logit_all_path.transpose(1,0,2)
    elif metric == 'min':
        output = torch.min(logit_all_path,axis=0)
    elif metric == 'median': 
        output = torch.median(logit_all_path,axis=0)
    return output

def output_final_step_tensor_v2(truth_ll, output_origin, predicted_label, ground_label):
    # output: final_correct, correct
    # accuracy, involving final step
    for i in range(len(truth_ll["logit"])): # list of list of [bsize, nClass]
    # total_logit += truth_ll["logit"][i][-1]
        truth_ll["logit"][i][-1]=np.array(truth_ll["logit"][i][-1])
    
    logit_all_path = np.array(truth_ll["logit"]).transpose(2,1,0,3)
    logit_all_path = np.squeeze(logit_all_path)
    logit_all_path = logit_all_path.transpose(1,0,2)
    output_origin = np.expand_dims(output_origin,0)
    logit_all_path = np.concatenate((logit_all_path,output_origin))
    logit_all_path = logit_all_path.transpose(1,0,2)
  
    if ground_label is not None:
        all_labels = np.array(ground_label)
        # all_labels = np.repeat(all_labels,logit_all_path.shape[1]).reshape(logit_all_path.shape[0],logit_all_path.shape[1])
        predicted_labels = np.array(predicted_label)
        # all_labels = np.repeat(predicted_labels,logit_all_path.shape[1]).reshape(logit_all_path.shape[0],logit_all_path.shape[1])
        return logit_all_path, predicted_labels, all_labels
    else:
        predicted_labels = np.array(predicted_labels)
        # all_labels = np.repeat(predicted_labels,logit_all_path.shape[1]).reshape(logit_all_path.shape[0],logit_all_path.shape[1])
        return logit_all_path, predicted_labels



def output_final_step_tensor_v2_direct_mode3(truth_ll, output_origin, predicted_label, ground_label):
    # output: final_correct, correct
    # accuracy, involving final step
    truth_ll["logit"].append([output_origin])
    for i in range(len(truth_ll["logit"])): # list of list of [bsize, nClass]
    # total_logit += truth_ll["logit"][i][-1]
        truth_ll["logit"][i][-1]=np.array(truth_ll["logit"][i][-1])

    logit_all_path = np.array(truth_ll["logit"]).transpose(2,1,0,3)
    logit_all_path = np.squeeze(logit_all_path)

    output=[]
    if len(predicted_label)>1:
        for i in range(len(predicted_label)):
            out = logit_all_path[i][np.argmax(logit_all_path[i],axis=1)==predicted_label[i].item()]
            output.append(out[0])
        output = np.array(output)
        output = torch.tensor(output).cuda()
        output1 = torch.tensor(output_origin).cuda()
    else:
        output = logit_all_path[np.argmax(logit_all_path,axis=1)==predicted_label.item()]
        output = torch.tensor([output[0]]).cuda()
        output1 = torch.tensor(output_origin).cuda()
    return output