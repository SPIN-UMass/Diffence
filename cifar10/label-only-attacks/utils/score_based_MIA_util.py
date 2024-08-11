import numpy as np
import math
import tensorflow as tf
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, auc



class black_box_benchmarks(object):
    
    def __init__(self, shadow_train_performance, shadow_test_performance, 
                 target_train_performance, target_test_performance, num_classes):
        '''
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels. 
        '''
        self.num_classes = num_classes
        
        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance
        
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)==self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)==self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)==self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)==self.t_te_labels).astype(int)
        
        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])
        
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)
        
        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)

        self.s_tr_loss = self._loss(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_loss = self._loss(self.s_te_outputs, self.s_te_labels)
        self.t_tr_loss = self._loss(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_loss = self._loss(self.t_te_outputs, self.t_te_labels)

        self.m_true = np.r_[ np.ones( len(self.t_tr_labels)  ), np.zeros( len(self.t_te_labels) ) ]
        
        self.s_true = np.r_[ np.ones( len(self.s_tr_labels)  ), np.zeros( len(self.s_te_labels) ) ]
        self.t_true = np.r_[ np.ones( len(self.t_tr_labels)  ), np.zeros( len(self.t_te_labels) ) ]
        
    
    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))
    
    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)),axis=1)
    
    def _loss(self, preds, y):
        criterion = nn.CrossEntropyLoss(reduction='none') 
        one_hot = np.zeros( (len(y), self.num_classes) )
        for i in range( len(y) ):
            one_hot[i] = tf.keras.utils.to_categorical( y[i], num_classes=self.num_classes) 
        loss = np.asarray([-math.log(y_pred) if y_pred > 0 else y_pred+1e-50 for y_pred in preds[one_hot.astype(bool) ] ])
        return loss
    
    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)
    
    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values>=value)/(len(tr_values)+0.0)
            te_ratio = np.sum(te_values<value)/(len(te_values)+0.0)
            acc = 0.5*(tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc 
        return thre
    
    def _mem_inf_via_corr(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        t_tr_acc = np.sum(self.t_tr_corr)/(len(self.t_tr_corr)+0.0)
        t_te_acc = np.sum(self.t_te_corr)/(len(self.t_te_corr)+0.0)
        mem_inf_acc = 0.5*(t_tr_acc + 1 - t_te_acc)
        #print('For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc) )
        return
    
    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0

        "use class-dependent threshold for MIA"
        class_dependent_thresholds = []
        for num in range(self.num_classes):
            thre = self._thre_setting(s_tr_values[self.s_tr_labels==num], s_te_values[self.s_te_labels==num])
            class_dependent_thresholds.append(thre)
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels==num]>=thre)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels==num]<thre)

        mem_inf_acc = 0.5*(t_tr_mem/(len(self.t_tr_labels)+0.0) + t_te_non_mem/(len(self.t_te_labels)+0.0))


        m_pred = np.zeros( len(self.t_tr_labels) + len(self.t_te_labels) )


        for i in range( len(self.t_tr_labels)): 
            m_pred[i] = t_tr_values[i] >= class_dependent_thresholds[ self.t_tr_labels[i] ]

        for i in range( len(self.t_te_labels) ):
            index = i + len(self.t_tr_labels)
            m_pred[index] = t_te_values[i] >= class_dependent_thresholds[ self.t_te_labels[i] ]


        pred_label = m_pred
        eval_label = self.m_true
        '''
        print('For membership inference attack via {n}, the attack acc is {acc:.3f}'.format(n=v_name,acc=mem_inf_acc))
        print("Accuracy: %.4f | Precision %.4f | Recall %.4f | f1_score %.4f" % ( accuracy_score(eval_label, pred_label), precision_score(eval_label,pred_label),\
                                    recall_score(eval_label,pred_label), f1_score(eval_label,pred_label)))
        print()
        print()
        '''
        return
        
    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]):
        if (all_methods) or ('correctness' in benchmark_methods):
            self._mem_inf_via_corr()
        if (all_methods) or ('confidence' in benchmark_methods):
            self._mem_inf_thre('confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
        if (all_methods) or ('entropy' in benchmark_methods):
            self._mem_inf_thre('entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
        if (all_methods) or ('modified entropy' in benchmark_methods):
            self._mem_inf_thre('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr)

        return

    # from https://github.com/cchoquette/membership-inference/blob/main/attacks.py
    def _get_max_accuracy(self, y_true, probs, thresholds=None,ratio=0.5):
        """Return the max accuracy possible given the correct labels and guesses. Will try all thresholds unless passed.
        Args:
            y_true: True label of `in' or `out' (member or non-member, 1/0)
            probs: The scalar to threshold
            thresholds: In a blackbox setup with a shadow/source model, the threshold obtained by the source model can be passed
            here for attackin the target model. This threshold will then be used.

        Returns: max accuracy possible, accuracy at the threshold passed (if one was passed), the max precision possible,
        and the precision at the threshold passed.
        """
        if thresholds is None:
            fpr, tpr, thresholds = roc_curve(y_true, probs)

        accuracy_scores = []
        precision_scores = []
        ratios=[]
        for thresh in thresholds:
            predicted_result = [1 if m > thresh else 0 for m in probs]
            ratios.append(np.sum(predicted_result)/len(probs))
            accuracy_scores.append(accuracy_score(y_true,
                                                predicted_result))
            precision_scores.append(precision_score(y_true, predicted_result))

        accuracies = np.array(accuracy_scores)
        precisions = np.array(precision_scores)
        max_accuracy = accuracies.max()
        max_precision = precisions.max()
        max_accuracy_threshold = thresholds[accuracies.argmax()]
        max_precision_threshold = thresholds[precisions.argmax()]
        ratios=np.abs(np.array(ratios)-ratio)
        accuracy_ratio = accuracies[ratios.argmin()]
        precision_ratio = precisions[ratios.argmin()]
        thre_ratio = thresholds[ratios.argmin()]

        return max_accuracy, max_accuracy_threshold, max_precision, max_precision_threshold, accuracy_ratio, precision_ratio,thre_ratio

    # from https://github.com/cchoquette/membership-inference/blob/main/attacks.py
    def _get_threshold(self, source_m, source_stats, target_m, target_stats):
        """ Train a threshold attack model and get teh accuracy on source and target models.

        Args:
            source_m: membership labels for source dataset (1 for member, 0 for non-member)
            source_stats: scalar values to threshold (attack features) for source dataset
            target_m: membership labels for target dataset (1 for member, 0 for non-member)
            target_stats: scalar values to threshold (attack features) for target dataset

        Returns: best acc from source thresh, precision @ same threshold, threshold for best acc,
            precision at the best threshold for precision. all tuned on source model.

        """
        # find best threshold on source data
        acc_source, t, prec_source, tprec,   ratio_acc_source, ratio_prec_source,ratio_thre_source  = self._get_max_accuracy(source_m, source_stats)

        # find best accuracy on test data (just to check how much we overfit)
        acc_test, _, prec_test, _,    ratio_acc_test, ratio_prec_test,ratio_thre_test = self._get_max_accuracy(target_m, target_stats)

        # get the test accuracy at the threshold selected on the source data
        acc_test_t, _, _, _,_,_,_ = self._get_max_accuracy(target_m, target_stats, thresholds=[t])
        _, _, prec_test_t, _ ,_,_,_= self._get_max_accuracy(target_m, target_stats, thresholds=[tprec])

        ratio_acc_test_t, _, ratio_prec_test_t, _ ,_,_,_= self._get_max_accuracy(target_m, target_stats, thresholds=[ratio_thre_source])

        print(print(np.sum(target_stats[:len(target_stats)//2]>tprec*1e06),np.sum(target_stats[len(target_stats)//2:]>tprec*1e-6)))
        print("acc src: {}, acc test (best thresh): {}, acc test (src thresh): {}, thresh: {}".format(acc_source, acc_test,
                                                                                                        acc_test_t, t))
        print(
            "prec src: {}, prec test (best thresh): {}, prec test (src thresh): {}, thresh: {}".format(prec_source, prec_test,
                                                                                                    prec_test_t, tprec))
        
        print(
            "ratio| acc test (ratio thresh): {}, prec test (ratio thresh): {}, thresh: {}".format(ratio_acc_test_t,  ratio_prec_test_t,ratio_thre_source
                                                                                                    ))
        return acc_test_t, prec_test_t, t, tprec
    
    @staticmethod
    def _get_max_accuracy_static( y_true, probs, thresholds=None):
        """Return the max accuracy possible given the correct labels and guesses. Will try all thresholds unless passed.
        Args:
            y_true: True label of `in' or `out' (member or non-member, 1/0)
            probs: The scalar to threshold
            thresholds: In a blackbox setup with a shadow/source model, the threshold obtained by the source model can be passed
            here for attackin the target model. This threshold will then be used.

        Returns: max accuracy possible, accuracy at the threshold passed (if one was passed), the max precision possible,
        and the precision at the threshold passed.
        """
        if thresholds is None:
            fpr, tpr, thresholds = roc_curve(y_true, probs)
        else:
            fpr, tpr, _ = roc_curve(y_true, probs)

        max_auc = auc(fpr, tpr)
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        ratios=[]
        for thresh in thresholds:
            predicted_result = [1 if m >= thresh else 0 for m in probs]
            ratios.append(np.sum(predicted_result)/len(probs))
            accuracy_scores.append(accuracy_score(y_true,
                                                predicted_result))
            precision_scores.append(precision_score(y_true, predicted_result))
            recall_scores.append(recall_score(y_true, predicted_result))
            f1_scores.append(f1_score(y_true,predicted_result))

        accuracies = np.array(accuracy_scores)
        precisions = np.array(precision_scores)
        recalls = np.array(recall_scores)
        f1_scores = np.array(f1_scores)
        
        max_accuracy = accuracies.max()
        print("************")
        print(accuracies)
        print(1-(fpr+(1-tpr))/2)
        print("******************")


        max_accuracy = np.max(1-(fpr+(1-tpr))/2)
        max_precision = precisions.max()
        max_accuracy_threshold = thresholds[accuracies.argmax()]
        max_precision_threshold = thresholds[precisions.argmax()]


        return max_accuracy, max_auc, max_accuracy_threshold, max_precision, max_precision_threshold, precisions, recalls, f1_scores, tpr, fpr