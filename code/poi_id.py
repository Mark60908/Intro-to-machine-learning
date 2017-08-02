#!/usr/bin/python
from __future__ import division
import sys
import pickle
import pandas as pd
import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from sklearn.grid_search import GridSearchCV
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from tester import test_classifier
sys.path.append("../tools/")

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
### Create pandas dataframe
df = pd.DataFrame(data_dict)

### Clean data
df = df.replace("NaN", 0)



### Task 2: Remove outliers
#print df.head()
#print np.argmax(df.bonus)
#df = df.drop(130)

df = df.drop('TOTAL', axis=1)
### Task 3: Create new feature(s)
df = df.T
df['wealth'] = df["salary"] + df["bonus"] + df["long_term_incentive"] + df["restricted_stock"] + df["other"]
df['proportion_from_poi_mail'] = np.float64(df["from_poi_to_this_person"])/np.float64(df["to_messages"])
df['proportion_to_poi_mail'] = np.float64(df["from_this_person_to_poi"]) / np.float64(df["from_messages"])
df['wealth'].fillna(0, inplace=True)
df['proportion_from_poi_mail'].fillna(0, inplace=True)
df['proportion_to_poi_mail'].fillna(0, inplace=True)


### Store to my_dataset for easy export below.
my_dataset = df.T.to_dict()



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_to_test = ['poi','bonus','salary','deferral_payments','deferred_income','director_fees',
                    'exercised_stock_options','expenses','total_payments','total_stock_value',
                    'from_messages','from_poi_to_this_person','from_this_person_to_poi',
                    'loan_advances','long_term_incentive','other','restricted_stock',
                    'restricted_stock_deferred','shared_receipt_with_poi','to_messages', 
                    'proportion_to_poi_mail', 'proportion_from_poi_mail', 'wealth']

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from tester import test_classifier

combined_features = FeatureUnion([('scale', MinMaxScaler())])
estimators = [('features', combined_features),('classify', DecisionTreeClassifier())]
pclf = Pipeline(estimators)
test_classifier(pclf, my_dataset, features_to_test)
importances = pclf.steps[-1][1].feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(22):
    print("%d. %s (%0.2f)" % (f + 1, features_to_test[1:][indices[f]], importances[indices[f]]))


### Selecing features based on the importance metrics
features_list = ['poi', 'bonus', 'total_stock_value', 'shared_receipt_with_poi',
                  'long_term_incentive', 'deferred_income',
                 'exercised_stock_options', 'total_payments', 'proportion_to_poi_mail']



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


def decision_tree(my_dataset, features_list):
    
    from sklearn.tree import DecisionTreeClassifier
    combined_features = FeatureUnion([('scale', MinMaxScaler())])
    estimators = [('features', combined_features),('classify', DecisionTreeClassifier())]
    pclf = Pipeline(estimators)
    test_classifier(pclf, my_dataset, features_list)

   
def gnb(my_dataset, features_list):
    
    from sklearn.naive_bayes import GaussianNB
    combined_features = FeatureUnion([('scale', MinMaxScaler())])
    estimators_2 = [('features', combined_features),('classify', GaussianNB())]
    pclf_2 = Pipeline(estimators_2)
    return test_classifier(pclf_2, my_dataset, features_list)


def random_forest(mydataset, features_list):

    from sklearn.ensemble import RandomForestClassifier
    combined_features = FeatureUnion([('scale', MinMaxScaler())])
    estimators_3 = [('features', combined_features),('classify', RandomForestClassifier())]
    pclf_3 = Pipeline(estimators_3)
    return test_classifier(pclf_3, my_dataset, features_list)


def knn(my_dataset, features_to_test):

    from sklearn.neighbors import KNeighborsClassifier
    combined_features = FeatureUnion([('scale', MinMaxScaler())])
    estimators_4 = [('features', combined_features),('classify', KNeighborsClassifier())]
    pclf_4 = Pipeline(estimators_4)
    return test_classifier(pclf_4, my_dataset, features_list)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


def tune_decision_tree():

    from sklearn.grid_search import GridSearchCV
    from sklearn.tree import DecisionTreeClassifier
    parameters = {'criterion':('gini', 'entropy'), 
             'splitter':('best','random')}
    pclf_1 = GridSearchCV(DecisionTreeClassifier(), param_grid = parameters,scoring = 'recall')
    pclf_1.get_params()

def tune_knn():
    
    from sklearn.neighbors import KNeighborsClassifier
    parameters = {'n_neighbors':[1,10], 'weight':['uniform', 'distance'], 'algorithm':('ball_tree','kd_tree','brute','auto'),
             'leaf_size': [0, 50], 'p': [1, 10], 'n_jobs': [0, 10]}
    pcl_3 = GridSearchCV(KNeighborsClassifier(), param_grid = parameters,scoring = 'recall')
    pcl_3.get_params()


def tune_random_forest():

    from sklearn.ensemble import RandomForestClassifier
    parameters = {'criterion':('gini', 'entropy'), "min_samples_split": [0, 10], "verbose": [0, 10],
             'n_estimators':[0, 30], "class_weight": ("balanced", None)}
    pcl_4 = GridSearchCV(RandomForestClassifier(), param_grid = parameters,scoring = 'recall')
    pcl_4.get_params()


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

def cross_validation():
    import sklearn.metrics
    from sklearn.cross_validation import train_test_split                          
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    #from sklearn.cross_validation import KFold
    from sklearn.model_selection import StratifiedKFold #New adding
    precisions = [] #New adding
    recalls = [] #New adding
    scores = []
    #skf=KFold(len(labels),3) 
    skf = StratifiedKFold(n_splits=3) #new adding
    for train_indices, test_indices in skf.split(features, labels):
        features_train= [features[ii] for ii in train_indices]
        features_test= [features[ii] for ii in test_indices]
        labels_train=[labels[ii] for ii in train_indices]
        labels_test=[labels[ii] for ii in test_indices]
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        clf.fit(features_train,labels_train)
        pred= clf.predict(features_test)
        scores.append(sklearn.metrics.accuracy_score(labels_test, pred))
        precisions.append(sklearn.metrics.precision_score(labels_test, pred))
        recalls.append(sklearn.metrics.recall_score(labels_test, pred))
    
    print "accuracy:", np.mean(scores)
    print "precision:", np.mean(precisions)
    print "recall:", np.mean(recalls)
    
gnb(my_dataset, features_list)
cross_validation()

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
dump_classifier_and_data(clf, my_dataset, features_list)

