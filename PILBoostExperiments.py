#this is code for experiments in ``Being Properly Improper''
#
#
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split 
import math
from numpy.random import default_rng
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
rng = default_rng()
import random
import unicodedata
import sys
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
import time

#important constants
eps = 1e-7
num_runs = 10
iterations = 1000
max_depf = 3

#this function is used to generate Bernoulli coin flips(fair/unfair) when needed
def flip(p):
    return 1 if random.random() < p else 0

###############################################################################
#this is a build of AdaBoost from scratch. The implementation of PILBoost comes from this
class AdaBoost:
    """ AdaBoost enemble classifier from scratch """

    def __init__(self):
        self.stumps = None
        self.stump_weights = None
        self.errors = None
        self.sample_weights = None

    def _check_X_y(self, X, y):
        """ Validate assumptions about format of input data"""
        assert set(y) == {-1, 1}, 'Expecting response variable to be formatted as ±1'
        return X, y
    
    def fit(self, X: np.ndarray, y: np.ndarray, iters: int):
        """ Fit the model using training data """

        X, y = self._check_X_y(X, y)
        n = X.shape[0]
    

        # init numpy arrays
        self.sample_weights = np.zeros(shape=(iters, n))
        self.stumps = np.zeros(shape=iters, dtype=object)
        self.stump_weights = np.zeros(shape=iters)
        self.errors = np.zeros(shape=iters)

        # initialize weights uniformly
        self.sample_weights[0] = np.ones(shape=n) / n

        for t in range(iters):
            curr_sample_weights = self.sample_weights[t]
            stump = DecisionTreeRegressor(max_depth=max_depf)
            stump = stump.fit(X, y, sample_weight=curr_sample_weights)

            # calculate error and stump weight from weak learner prediction
            stump_pred = stump.predict(X)
            err = curr_sample_weights[(np.sign(stump_pred) != y)].sum()
            stump_weight = np.log((1 - err + eps) / err + eps) / 2
            new_sample_weights = curr_sample_weights * np.exp(-stump_weight * y * stump_pred)
            new_sample_weights /= new_sample_weights.sum()

            # If not final iteration, update sample weights for t+1
            if t+1 < iters:
                self.sample_weights[t+1] = new_sample_weights

            # save results of iteration
            self.stumps[t] = stump
            self.stump_weights[t] = stump_weight
            self.errors[t] = err
    
        

        return self

    def predict(self, X):
        """ Make predictions using already fitted model """
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        return np.sign(np.dot(self.stump_weights, stump_preds))
    
###############################################################################
#this is the implementation of PILBoost
#It takes two hyperparameters: alpha and a_f
class PILBoost:
    
    def __init__(self):
        self.weak_learners = None
        self.weak_learner_weights = None
        self.errors = None
        self.functions = None

    def _check_X_y(self, X, y):
        """ Validate assumptions about format of input data"""
        assert set(y) == {-1, 1}, 'Expecting response variable to be formatted as ±1'
        return X, y
    
    def fit(self, X: np.ndarray, y: np.ndarray, iters: int, alpha, af):
        """ Fit the model using training data """

        #constants frequently used in computation of tilde f below
        #####
        frac0 = alpha/(alpha-1)
        
        frac1 = 1/(alpha-1)
        
        frac2 = 1/(alpha)
        
        frac3 = (alpha/(alpha-1))**(alpha/(alpha-1))
        ####
        
        X, y = self._check_X_y(X, y)
        n = X.shape[0]
        #epsilon = 1e-7
#        feature_importance = np.zeros(X.shape[1])
        # init numpy arrays
        self.weak_learners = np.zeros(shape=iters+1, dtype=object)
        self.weak_learner_weights = np.zeros(shape=iters+1)
        self.errors = np.zeros(shape=iters+1)
        self.functions = np.zeros(shape=(iters+1, n))

        # initialize functions to zero
        self.functions[0] = np.zeros(shape=n)
        sample_weights = np.zeros(shape = n)

        for t in range(1, iters+1):
            # update weights
            
            z = -y * self.functions[t-1]
            
            for i in range(1,n):
                if z[i]<= -frac0:
                    sample_weights[i] = 0
                elif -frac0 < z[i] <= 0:
                    sample_weights[i] = (frac0 + z[i])**(frac1)/((frac0 + z[i])**(frac1) + (2*frac3 - (frac0 + z[i])**(frac0))**(frac2))
                elif 0 < z[i] <= frac0:
                    sample_weights[i] = ((2*frac3 - (frac0 - z[i])**(frac0))**(frac2))/((frac0 - z[i])**frac1 + (2*frac3 - (frac0 - z[i])**frac0)**(frac2) )
                else:
                    sample_weights[i] = 1
            
            
            # fit weak learner using weights
            weak_learner = DecisionTreeRegressor(max_depth=max_depf)
            weak_learner =  weak_learner.fit(X, y, sample_weight=sample_weights)
            
            # calculate error and stump weight from weak learner prediction
            weak_learner_pred = weak_learner.predict(X)
            err = np.dot(sample_weights,np.multiply(weak_learner_pred,y))/n
            weak_learner_weight = af*err
#            feature_importance += weak_learner_weight*weak_learner.feature_importances_
            
            # update functions
            self.functions[t] = self.functions[t-1] + np.dot(weak_learner_weight, weak_learner_pred)

            # save results of iteration
            self.weak_learners[t] = weak_learner
            self.weak_learner_weights[t] = weak_learner_weight
            self.errors[t] = err
            
        #this code is useful for generating feature importance plots    
        #pyplot.bar(range(len(feature_importance)), feature_importance/feature_importance.sum())
        #pyplot.title('Insider Twister - us')
        #pyplot.title('No Twister - us')
        #pyplot.ylabel('Normalized Value')
        #pyplot.xlabel('Features')
        #pyplot.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        #pyplot.ylim([0, 0.5])
        #pyplot.savefig('Insider Twister - us.png', dpi=1000)
        #pyplot.savefig('Insider Twister - us.png', dpi=1000)
        #pyplot.show()

        
        return self
    
    def predict(self, X):
        """ Make predictions using already fitted model """
        weak_learners_temp = self.weak_learners[1:]
        weak_learners = weak_learners_temp[weak_learners_temp != 0]
        weak_learner_preds = np.array([weak_learner.predict(X) for weak_learner in weak_learners])
        weak_learner_weights_temp = self.weak_learner_weights[1:]
        weak_learners_weights = weak_learner_weights_temp[weak_learner_weights_temp != 0]
        classifier = np.dot(weak_learners_weights, weak_learner_preds)
        return np.sign(classifier), classifier, self.weak_learner_weights[1:]

##########################################################



####################################################################
# this function is used for model evaluation 
def evaluate_model(X, y, model, name):
    scores = np.zeros(num_runs)
    for j in range(num_runs):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y)

##       this adds label noise
#        train_labels_count = y_train.shape[0]
#        labels_to_flip_count = math.ceil(train_labels_count*0.0) #the fraction in parentheses indicates the percentage of training label noise
#        labels_to_flip = rng.choice(train_labels_count, size=labels_to_flip_count, replace=False)
#        for label in labels_to_flip:
#            if y_train[label] == -1:
#                y_train[label] = 1
#            else:
#                y_train[label] = -1
            

        #Insider Twister
#        X_train[:,8] = X_train[:,8] + np.random.normal(0, 60, X_train.shape[0])
#        train_samples_count = y_train.shape[0]
#        samples_to_twist_count = math.ceil(train_samples_count*1.0)
#        samples_to_twist = rng.choice(train_samples_count, size=samples_to_twist_count, replace=False)
#        for l in samples_to_twist:
#            X_train[l,15] =  (X_train[l,15] + random.randint(0, 1)) % 3
#            X_train[l,10] =  (X_train[l,10] + random.randint(0, 1)) % 10
####
        
        #feature noise for xd6 
#        for i in range(len(y_train)):
#            if flip(0.5) == 1:
#                for k in range(9):
#                    if flip(0.5) == 1:
#                        X_train[i][k] = not(X_train[i][k])
        
        #train and test model
        #t0 = time.time()
        if name == 'AdaBoost':
            clf = model.fit(X_train, y_train,iters=iterations)
            y_pred = clf.predict(X_test)
        elif name == first:    
            clf = model.fit(X_train, y_train, iters=iterations,alpha=1.1,af = 7)
            y_pred = clf.predict(X_test)[0]
        elif name == 'XGBoost':
            y_train[y_train == -1] = 0
            y_test[y_test == -1] = 0
            #model.fit(X_train, y_train, verbose=False, early_stopping_rounds=10, eval_metric='aucpr', eval_set=[(X_test, y_test)])
            model.fit(X_train, y_train, verbose=False)
            #this is useful for generating feature importance plots for XGBoost
            #pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
            #print(model)
            #pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
            #pyplot.title('Insider Twister - XGBoost')
            #pyplot.title('Insider Twister - XGBoost')
            #pyplot.ylabel('Normalized Value')
            #pyplot.xlabel('Features')
            #pyplot.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            #pyplot.ylim([0, 0.5])
            #pyplot.savefig('Insider Twister - XGBoost.png', dpi=1000)
            #pyplot.savefig('Insider Twister - XGBoost (Early Stopped 2).png', dpi=1000)
            #pyplot.show()
            y_pred = model.predict(X_test)
        elif name == second:    
            clf = model.fit(X_train, y_train, iters=iterations,alpha=2,af = 8)
            y_pred = clf.predict(X_test)[0]
            
        elif name == third:    
            clf = model.fit(X_train, y_train, iters=iterations,alpha=4,af = 15)
            y_pred = clf.predict(X_test)[0]
        else:
            clf = model.fit(X_train, y_train, iters=iterations,alpha=4,af = 15)
            y_pred = clf.predict(X_test)[0]
        #t1 = time.time()
        #print('{}'.format(names[i]), t1-t0)
        scores[j] = (y_pred == y_test).mean()
        #for tallying computation time
#        if name == 'AdaBoost':
#            computation_time_AdaBoost[j] = t1-t0
#        elif name == 'XGBoost':
#            computation_time_XGBoost[j] = t1-t0
#        elif name == first:
#            computation_time_PILBoost_1[j] = t1-t0
#        elif name == second:
#            computation_time_PILBoost_2[j] = t1-t0
#        elif name == third:
#            computation_time_PILBoost_3[j] = t1-t0
#        else:
#            print('done')
           
        

       # 
    
    return scores

first = '{} = 1.1'.format(unicodedata.lookup("GREEK SMALL LETTER ALPHA"))
second = '{} = 2'.format(unicodedata.lookup("GREEK SMALL LETTER ALPHA"))
third = '{} = 4'.format(unicodedata.lookup("GREEK SMALL LETTER ALPHA"))

#this function keeps track of the models and scores
def get_models():
    models, names = list(), list()
    #
    models.append(AdaBoost())
    names.append('AdaBoost')
    #
    models.append(PILBoost())
    names.append(first)
    #
    models.append(PILBoost())
    names.append(second)
    #
    models.append(PILBoost())
    names.append(third)
    #
    models.append(xgb.XGBClassifier(objective='binary:logistic', 
                            eval_metric="logloss", ## this avoids a warning...
                            missing=None, n_estimators=iterations, max_depth = max_depf, use_label_encoder=False))
    names.append('XGBoost')
    return models, names

#Datasets
#########################################

#Breast Cancer dataset
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
y[y == 0] = -1

#############################

#xd6 dataset
#data = pd.read_csv('xd6.csv')
#y = data.iloc[:,9].to_numpy()
#X = data.iloc[:,:9].to_numpy()
#y[y == 0] = -1
    
#############################
#pima indians diabetes dataset
#data = pd.read_csv('pima-indians-diabetes.csv')
#y = data.iloc[:,8].to_numpy()
#X = data.iloc[:,:8].to_numpy()
#y[y == 0] = -1

############################
#Online shopping dataset
#preprocessing code from https://www.kaggle.com/daewoongjun/how-can-we-convince-more-customers-to-buy
#data = pd.read_csv("online_shoppers_intention.csv")
#null_table = pd.DataFrame(data.isnull().sum().values.reshape(1,-1), columns = data.isnull().sum().index)
#null_table = null_table.rename(index = {0:'Total Null Values'})
#null_pct = null_table.iloc[0,:]/12330 *100
#null_pct = pd.DataFrame(null_pct.values.reshape(1,-1), columns = null_pct.index)
#null_pct = null_pct.rename(index = {0 : 'Null %'})
#null_table = null_table.append(null_pct)
#data = data.dropna()
#data = data.drop(data[data['Administrative_Duration'] < 0].index)
#data = data.drop(data[data['Informational_Duration'] < 0].index)
#data = data.drop(data[data['ProductRelated_Duration'] < 0].index)
#le = LabelEncoder()
#data['Month'] = le.fit_transform(data['Month'])
#data['VisitorType'] = le.fit_transform(data['VisitorType'])
#data['Weekend'] = le.fit_transform(data['Weekend'])
#data['Revenue'] = le.fit_transform(data['Revenue'])
#X = data.drop('Revenue',axis=1).to_numpy()
#y = data['Revenue'].to_numpy()
#y[y == 0] = -1
##########################
#Use this code for (insider twisted) one run experiments so that both models have exactly the same split
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y)
##X_sneaky = np.copy(X_train)
##
###Insider Twister
#X_train[:,8] = X_train[:,8] + np.random.normal(0, 60, X_train.shape[0])
##
#train_samples_count = y_train.shape[0]
#samples_to_twist_count = math.ceil(train_samples_count*1.0)
#samples_to_twist = rng.choice(train_samples_count, size=samples_to_twist_count, replace=False)
#for l in samples_to_twist:
#    X_train[l,15] =  (X_train[l,15] + random.randint(0, 1)) % 3
#    X_train[l,10] =  (X_train[l,10] + random.randint(0, 1)) % 10
##    if y_train[l] == -1:
##        y_train[l] = 1
##    else:
##        y_train[l] = -1
#
#print((X_sneaky[:,15] == X_train[:,15]).mean())
#print((X_sneaky[:,10] == X_train[:,10]).mean())


####################################################

# define models
models, names = get_models()
results = list()
#used for computation time
#computation_time_AdaBoost = np.zeros(num_runs)
#computation_time_PILBoost_1 = np.zeros(num_runs)
#computation_time_PILBoost_2 = np.zeros(num_runs)
#computation_time_PILBoost_3 = np.zeros(num_runs)
#computation_time_XGBoost = np.zeros(num_runs)

# evaluate each model
for i in range(len(models)):
	scores = evaluate_model(X, y, models[i], names[i])
	results.append(scores)
	# summarize performance
	print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
    
#figure size     
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

pyplot.rc('font', size=SMALL_SIZE)          # controls default text sizes
pyplot.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
pyplot.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
pyplot.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
pyplot.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
pyplot.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
pyplot.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title    
    
    
pyplot.boxplot(results, labels=names, showmeans=True)
str = 'Experiments'
pyplot.title(str)
pyplot.savefig('{}_test.png'.format(str), dpi=1000)
pyplot.show()
#print(scores)

#writing to text files for storage
with open('{}_test.txt'.format(str), 'w') as f:
    for i in range(len(models)):
        print(names[i], file=f)
        print(results[i], file=f)
        
#print('Average AdaBoost Compute Time')
#print(mean(computation_time_AdaBoost))
#print('Average PILBoost_1 Compute Time')
#print(mean(computation_time_PILBoost_1))
#print('Average PILBoost_2 Compute Time')
#print(mean(computation_time_PILBoost_2))
#print('Average PILBoost_3 Compute Time')
#print(mean(computation_time_PILBoost_3))
#print('Average XGBoost Compute Time')
#print(mean(computation_time_XGBoost))
        
        
#############################
#this is used for Welch's t-test
#from scipy import stats
#import numpy as np
#
#
#np.random.seed(12345678) # fix random seed to get same numbers
#us = [0.85104082, 0.84941876, 0.84833739, 0.84914842, 0.85158151, 0.85050014,
# 0.84995945, 0.84563396, 0.8477967, 0.85320357]
#other = [0.81914031, 0.8359016,  0.83698297, 0.79670181, 0.83130576, 0.80643417,
# 0.85077048, 0.83346851, 0.84022709, 0.84049743]
#b = stats.ttest_ind(us,other,equal_var=False)
#print(b)