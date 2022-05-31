#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score   
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from numpy.random import default_rng
rng = default_rng()
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.utils import shuffle
from sklearn.datasets import load_breast_cancer      
          
eps = 1e-2


def sigmoid(z):
    return 1/(1 + np.exp(-z))

#use these functions to make computation easier for PseudoMBoost. Also, easier to make sure equations are correct
def frac0(alpha):
    return alpha/(alpha-1)

def frac1(alpha):
    return 1/(alpha-1)

def frac2(alpha):
    return 1/(alpha)

def frac3(alpha):
    return (alpha/(alpha-1))**(alpha/(alpha-1))

def logit(x):
    return np.log(x/(1-x))

class PseudoMBoost:
    
    def __init__(self):
        self.stumps = None
        self.stump_weights = None
        self.errors = None
        self.functions = None

    def _check_X_y(self, X, y):
        """ Validate assumptions about format of input data"""
        assert set(y) == {-1, 1}, 'Expecting response variable to be formatted as ±1'
        return X, y
    
    def fit(self, X: np.ndarray, y: np.ndarray, iters: int, alpha, af):
        """ Fit the model using training data """

        X, y = self._check_X_y(X, y)
        n = X.shape[0]
        #epsilon = 1e-7
        #feature_importance = np.zeros(X.shape[1])
        # init numpy arrays
        self.stumps = np.zeros(shape=iters+1, dtype=object)
        self.stump_weights = np.zeros(shape=iters+1)
        self.errors = np.zeros(shape=iters+1)
        self.functions = np.zeros(shape=(iters+1, n))

        # initialize functions to zero
        self.functions[0] = np.zeros(shape=n)

        for t in range(1, iters+1):
            # update weights
            sample_weights = np.zeros(shape = n)
            
            z = -y * self.functions[t-1]
            
            for i in range(1,n):
                if z[i]<= -frac0(alpha):
                    sample_weights[i] = 0
                elif -frac0(alpha) < z[i] <= 0:
                    sample_weights[i] = (frac0(alpha) + z[i])**(frac1(alpha))/((frac0(alpha) + z[i])**(frac1(alpha)) + (2*frac3(alpha) - (frac0(alpha) + z[i])**(frac0(alpha)))**(frac2(alpha)))
                elif 0 < z[i] <= frac0(alpha):
                    sample_weights[i] = ((2*frac3(alpha) - (frac0(alpha) - z[i])**(frac0(alpha)))**frac2(alpha))/((frac0(alpha) - z[i])**frac1(alpha) + (2*frac3(alpha) - (frac0(alpha) - z[i])**frac0(alpha))**(frac2(alpha)) )
                else:
                    sample_weights[i] = 1
            
            
            # fit weak learner using weights
            #stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            stump = DecisionTreeRegressor(max_depth=max_depf)
            stump = stump.fit(X, y, sample_weight=sample_weights)
            
            # calculate error and stump weight from weak learner prediction
            stump_pred = stump.predict(X)
     
            #print(np.sum((stump_pred != y)))
            #err = sample_weights[(stump_pred != y)].sum()
            err = np.dot(sample_weights,np.multiply(stump_pred,y))/n

            stump_weight = af*err
            #feature_importance += stump_weight*stump.feature_importances_
            # update functions
            self.functions[t] = self.functions[t-1] + np.dot(stump_weight, stump_pred)

            # save results of iteration
            self.stumps[t] = stump
            self.stump_weights[t] = stump_weight
            self.errors[t] = err
        

        
        return self
    
    def predict(self, X):
        """ Make predictions using already fitted model """
        stumps_temp = self.stumps[1:]
        stumps = stumps_temp[stumps_temp != 0]
        #print(stumps)
        stump_preds = np.array([stump.predict(X) for stump in stumps])
        #print(stump_preds)
        #print(self.stump_weights[1:])
        stump_weights_temp = self.stump_weights[1:]
        stump_weights = stump_weights_temp[stump_weights_temp != 0]
        classifier = np.dot(stump_weights, stump_preds)
        #print(type(np.sign(classifier))
        return np.sign(classifier), classifier, self.stump_weights[1:]
    
 
    
#set to 1000 for full experiment
iterations = 10
#diabetes is 3
#cancer is 1
#xd6 is 3
max_depf = 1  

# Create dataset
#X, y = datasets.make_hastie_10_2(n_samples=1000, random_state=0) 
##alpha = 1.01 and af = 5 is good for hastie          
# Load dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
y[y == 0] = -1
#print(np.count_nonzero(y == 0))

#xd6 dataset
#data = pd.read_csv('xd6.csv')
#y = data.iloc[:,9].to_numpy()
#X = data.iloc[:,:9].to_numpy()
#y[y == 0] = -1

#pima indians diabetes
#data = pd.read_csv('pima-indians-diabetes.csv')
#y = data.iloc[:,8].to_numpy()
#X = data.iloc[:,:8].to_numpy()
#y[y == 0] = -1 
    
noises = [.05,.1,.15,.2,.25,.3,.35,.4,.45]
#noises = [.1,.2,.3]
num_noises = len(noises)    
acc_xgb = np.zeros(num_noises)    
acc_learned = np.zeros(num_noises) 
acc_learned_new = np.zeros(num_noises) 
acc_fixed = np.zeros(num_noises) 
acc_menon = np.zeros(num_noises) 

redundancy = 10


for noise in noises:
        # split the dataset into 70% training and 30% testing data
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y)
        print('Number of Test Samples')
        print(np.count_nonzero(y_test == -1) + np.count_nonzero(y_test == 1))
        print('Number of 1"s in test')
        print(np.count_nonzero(y_test == 1))
        train_labels_count = y_train.shape[0]
        labels_to_flip_count = math.ceil(train_labels_count*noise)
        labels_to_flip = rng.choice(train_labels_count, size=labels_to_flip_count, replace=False)
        
        print('Number of Train Samples')
        print(np.count_nonzero(y_train == -1) + np.count_nonzero(y_train == 1))
        
        print(f'Number of Labels we are flipping {labels_to_flip_count}')
        
        print('\n')
        
        print(f'NOISE LEVEL {noise}')
        
        for label in labels_to_flip:
            if (y_train[label] == 1):
                y_train[label] = -1
            else:
                y_train[label] = 1    
        
        
        print(' ')
        print(' ')
        
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0

        clf_xgb = xgb.XGBClassifier(objective='binary:logistic', 
                                    eval_metric="logloss", ## this avoids a warning...
                                    missing=None, n_estimators=iterations, max_depth = max_depf, use_label_encoder=False)
        
        clf_xgb.fit(X_train, 
                    y_train,
                    verbose=False, #)
                    ## the next three arguments set up early stopping.
                    #early_stopping_rounds=10,
                    eval_metric='aucpr',
                    eval_set=[(X_train, y_train)])
        
        y_pred = clf_xgb.predict(X_test)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        acc_xgb[noises.index(noise)] += accuracy
        print("XGBoost Accuracy:",accuracy)
        
        print(' ')
        xgb_confusion_matrix = confusion_matrix(y_test, predictions)
        #print(f'Confusion matrix = \n {xgb_confusion_matrix}')
        p_noise = np.mean([xgb_confusion_matrix[1][0]/(xgb_confusion_matrix[1][1]+xgb_confusion_matrix[1][0]),xgb_confusion_matrix[0][1]/(xgb_confusion_matrix[0][1]+xgb_confusion_matrix[0][0])])
        eta_c = (xgb_confusion_matrix[1][1] + xgb_confusion_matrix[1][0])/xgb_confusion_matrix.sum()
        eta_t = (xgb_confusion_matrix[0][1]+xgb_confusion_matrix[1][1])/xgb_confusion_matrix.sum()
        #print(f'Estimated label noise = {p_noise}')
        #print(f'Estimated clean posterior = {eta_c}')
        alpha_star = logit(eta_c)/logit(eta_c*(1-p_noise) + (1-eta_c)*p_noise)
        delta = .1
        alpha_star_new = 1 + ((1-2*eta_t)*delta)/((eta_t - 1)*eta_t*logit(eta_t))
        print(f'alpha* = {alpha_star}')
        print(f'alpha*_new = {alpha_star_new}')
        print(' ')
        
        #afs
        #xd6 
        #a_f_s = 8
        #breast cancer 
        a_f_s = 7
        #a_f_s = 1 #for high noise
        #diabetes
        #a_f_s = .1 #for diabetes
        
        y_train[y_train == 0] = -1
        y_test[y_test == 0] = -1
        clf = PseudoMBoost().fit(X_train, y_train, iters=iterations,alpha=alpha_star,af = a_f_s)
        y_pred = clf.predict(X_test)[0]
        test_err = (y_pred != y_test).mean()
        ##print(f'Test error: {test_err:.1%}')
        acc_learned[noises.index(noise)] += 1-test_err
        print("PseudoMBoost Learned Accuracy:",1-test_err)
        #print(confusion_matrix(y_test, y_pred))
        print(' ')
        
        clf = PseudoMBoost().fit(X_train, y_train, iters=iterations,alpha=alpha_star_new,af = a_f_s)
        y_pred = clf.predict(X_test)[0]
        test_err = (y_pred != y_test).mean()
        ##print(f'Test error: {test_err:.1%}')
        acc_learned_new[noises.index(noise)] += 1-test_err
        print("PseudoMBoost Taylor Series Accuracy:",1-test_err)
        #print(confusion_matrix(y_test, y_pred))
        
        
        print(' ')
        fix_alfa = 1.1
        clf = PseudoMBoost().fit(X_train, y_train, iters=iterations,alpha=fix_alfa,af = a_f_s)
        y_pred = clf.predict(X_test)[0]
        test_err = (y_pred != y_test).mean()
        acc_fixed[noises.index(noise)] += 1-test_err
        ##print(f'Test error: {test_err:.1%}')
        print("PseudoMBoost Fixed Accuracy:",1-test_err)
        print(confusion_matrix(y_test, y_pred))


        ##Menon's method to estimate noise, and thus alpha
        clf = tree.DecisionTreeClassifier(criterion = "entropy",min_samples_leaf=math.ceil(np.sqrt(len(X_train))),max_leaf_nodes=math.ceil(np.log(len(X_train))))
        clf = clf.fit(X_train, y_train)
        print(f'{clf.get_n_leaves()} Leaves')
        print(f'{len(X_train)} examples')
        
        
        leaf_outcomes = clf.apply(X_train)
        
        tree_record = {}
        tree_record = tree_record.fromkeys(leaf_outcomes)
        inter_list = [0, 0]
        for index in range(len(X_train)):
            if tree_record[leaf_outcomes[index]] == None:
                if y_train[index] == -1:
                    tree_record[leaf_outcomes[index]] = [0,1]
                else:
                    tree_record[leaf_outcomes[index]] = [1,1]
            else:
                inter_list = tree_record[leaf_outcomes[index]]
                
                if y_train[index] == -1:
                    inter_list[0] += 0
                    inter_list[1] += 1
                else:
                    inter_list[0] += 1
                    inter_list[1] += 1
                    
                tree_record[leaf_outcomes[index]] = inter_list
        
        eta_max = 0
        eta_min = 1
        temp_list = []
        eta_temp = 0
        eta_average = 0
        
        for entry in tree_record:
            temp_list = tree_record[entry]
            eta_temp = temp_list[0]/temp_list[1]
            eta_average += temp_list[0]/temp_list[1]
            
            if eta_temp < eta_min:
                eta_min = eta_temp
                
            if eta_temp > eta_max:
                eta_max = eta_temp
        
        print('Eta Min and Max')
        print(eta_min,eta_max)
        print('Noise Estimate')
        print(np.sqrt(eta_min*(1-eta_max)))
        
        p_noise_new = np.sqrt(eta_min*(1-eta_max))
        eta_average = eta_average/len(tree_record)
        print(f'Eta Average is {eta_average}')
        
        alpha_star_newest = logit((eta_average - p_noise_new)/(1-2*p_noise_new))/logit(eta_average) + eps
        print(alpha_star_newest)

        print(' ')
        clf = PseudoMBoost().fit(X_train, y_train, iters=iterations,alpha=alpha_star_newest,af = a_f_s)
        y_pred = clf.predict(X_test)[0]
        test_err = (y_pred != y_test).mean()
        acc_menon[noises.index(noise)] += 1-test_err
        ##print(f'Test error: {test_err:.1%}')
        print("PseudoMBoost Menon Accuracy:",1-test_err)
        #print(confusion_matrix(y_test, y_pred))



acc_xgb = acc_xgb/redundancy 
acc_learned = acc_learned/redundancy
acc_learned_new = acc_learned_new/redundancy
acc_fixed = acc_fixed/redundancy
acc_menon = acc_menon/redundancy

plt.plot(noises, acc_xgb,'k')
plt.plot(noises, acc_learned,'b')
plt.plot(noises, acc_learned_new,'y')
plt.plot(noises, acc_fixed,'g')
plt.plot(noises, acc_menon,'r')
plt.title(f'Adaptive Alpha Experiment')
plt.xlabel('Noise Level')
plt.legend(['XGBoost', 'Original Learned PILBoost', 'Taylor Series PILBoost' , f'Fixed PILBoost alpha = {fix_alfa}','Menon PILBoost'])
plt.savefig(f'cancer_w_{fix_alfa}.png', dpi=1000)
plt.show()