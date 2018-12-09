# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 10:17:51 2018

@author: saughosh
"""

"""
The overall parameters have been divided into 3 categories by XGBoost authors:

1.General Parameters: Guide the overall functioning
2.Booster Parameters: Guide the individual booster (tree/regression) at each step
3. Learning Task Parameters: Guide the optimization performed

1.General Parameters

These define the overall functionality of XGBoost.
 1.booster [default=gbtree]
   Select the type of model to run at each iteration. It has 2 options:
      gbtree: tree-based models
      gblinear: linear models
 
  2.silent [default=0]:
      Silent mode is activated is set to 1, i.e. no running messages will be printed.It’s generally good to keep it 0 as the messages might help in understanding the model.

  3.nthread [default to maximum number of threads available if not set] :This is used for parallel processing and number of cores in the system should be entered
  If you wish to run on all cores, value should not be entered and algorithm will detect automatically
  There are 2 more parameters which are set automatically by XGBoost and you need not worry about them. Lets move on to Booster parameters.

 

2.Booster Parameters :
    Though there are 2 types of boosters, I’ll consider only tree booster here because it always outperforms the linear booster and thus the later is rarely used.

    1:eta [default=0.3] :
        Analogous to learning rate in GBM
        Makes the model more robust by shrinking the weights on each step
        Typical final values to be used: 0.01-0.2
        
    2.min_child_weight [default=1] :
        1.Defines the minimum sum of weights of all observations required in a child.
        2.This is similar to min_child_leaf in GBM but not exactly. This refers to min “sum of weights” of observations while GBM has min “number of observations”.
        3.Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
          Too high values can lead to under-fitting hence, it should be tuned using CV.

    3.max_depth [default=6] :
        1.The maximum depth of a tree, same as GBM.
        2.Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
        3.Should be tuned using CV.
        4.Typical values: 3-10

    4.max_leaf_nodes :
        1.The maximum number of terminal nodes or leaves in a tree.
        2.Can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves.
        3.If this is defined, GBM will ignore max_depth.

    5.gamma [default=0] :
        1.A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
        2.Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.

    6.max_delta_step [default=0] :
        1.In maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative.
        2.Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.
          This is generally not used but you can explore further if you wish.

    7.subsample [default=1] :
        1.Same as the subsample of GBM. Denotes the fraction of observations to be randomly samples for each tree.
        2.Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
        3.Typical values: 0.5-1

    8.colsample_bytree [default=1] :
        1.Similar to max_features in GBM. Denotes the fraction of columns to be randomly samples for each tree.
        2.Typical values: 0.5-1

    9.colsample_bylevel [default=1]:
        1.Denotes the subsample ratio of columns for each split, in each level.
        2.I don’t use this often because subsample and colsample_bytree will do the job for you. but you can explore further if you feel so.

    10.lambda [default=1] :
        1.L2 regularization term on weights (analogous to Ridge regression)
        2.This used to handle the regularization part of XGBoost. Though many data scientists don’t use it often, it should be explored to reduce overfitting.

    11.alpha [default=0] :
        1.L1 regularization term on weight (analogous to Lasso regression)
        2.Can be used in case of very high dimensionality so that the algorithm runs faster when implemented

    12.scale_pos_weight [default=1] :
        1.A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence.
 

3.Learning Task Parameters :
    These parameters are used to define the optimization objective the metric to be calculated at each step.

    1.objective [default=reg:linear]:
        1.This defines the loss function to be minimized. Mostly used values are:
            binary:logistic –logistic regression for binary classification, returns predicted probability (not class)
            1.multi:softmax –multiclass classification using the softmax objective, returns predicted class (not probabilities)
            2.you also need to set an additional num_class (number of classes) parameter defining the number of unique classes
            3.multi:softprob –same as softmax, but returns predicted probability of each data point belonging to each class

    2.eval_metric [ default according to objective ] :
        The metric to be used for validation data.The default values are rmse for regression and error for classification.
        Typical values are:
            1.rmse – root mean square error
            2.mae – mean absolute error
            3.logloss – negative log-likelihood
            4.error – Binary classification error rate (0.5 threshold)
            5.merror – Multiclass classification error rate
            6.mlogloss – Multiclass logloss
            7.auc: Area under the curve

    3.seed [default=0] :
        The random number seed.Can be used for generating reproducible results and also for parameter tuning.
        If you’ve been using Scikit-Learn till now, these parameter names might not look familiar. A good news is that xgboost module in python has an sklearn wrapper called XGBClassifier. It uses sklearn style naming convention. The parameters names which will change are:
            1.eta –> learning_rate 
            2.lambda –> reg_lambda
            3.alpha –> reg_alpha

"""
#-------------------------------------------------------------
"""
Overfitting & Underfitting Notes
1.Dealing with overfitting :
    1.use less features 
    2.use more training samples
    3.increase regularization 

   In xgboost :
    1.reduce the depth of the tree
    2.increase the min child weight param
    3.add more randomness using subsample , colsample_bytree param
    4.increase lambda and alpha regularization param

2.Dealing with underfitting :
    1.add more features
    2.decrease regualization
    
    In xgboost :
    1.increase depth of the tree
    2.decrease min child weight param
    3.decrease gamma param
    4.decrease alpha , lamba param
    
"""

import pandas as pd 
import numpy  as np
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12,10

dataset = pd.read_csv(r"C:\Users\saughosh\Desktop\IMS\Course\DS3\6.Machine Learning\Algo\Classfication\Tree\Xboost\Model\Dataset\Model_Dataset\train.csv")
predictors = [d for d in dataset.columns if d not in ["Disbursed"]]
x_train, x_test, y_train,y_test = train_test_split(dataset[predictors], dataset["Disbursed"], test_size = .3 , random_state = 42)

###convert into xgb format 
d_train = xgb.DMatrix(x_train, label = y_train)
d_test  = xgb.DMatrix(x_test, label = y_test)


def prob_score(y_act,y_pred):
    model_metrics = []
    prb_range = np.linspace(0,1,100,endpoint=False)
    for p in prb_range:
        pred = y_pred > round(p,2)
        confusion_metrics =  metrics.confusion_matrix(y_act,pred)
        accuracy = metrics.accuracy_score(y_act,pred)
        precission = metrics.precision_score(y_act,pred)
        specitivity = confusion_metrics[0,0]/(confusion_metrics[0,0] + confusion_metrics[0,1])
        sensitivity = confusion_metrics[1,1]/(confusion_metrics[1,1] + confusion_metrics[1,0])
        f1 = metrics.f1_score(y_act,pred)
        base = np.sum(pred == True)
        metric = np.array([p,accuracy,precission,sensitivity,specitivity,f1,base])
        model_metrics.append(metric)

    model_metrics = round(pd.DataFrame(model_metrics),2)
    model_metrics.columns = [["Probablity","Accuracy","Precission","Sensitivity","Specificity","f1","Base"]]
    return model_metrics


def auc_score(y_act,y_pred):
    auc_score = metrics.roc_auc_score(y_act,y_pred)
    return auc_score

##building the modeling using xgb settings
def lift_score(dataset, pred , y):
    dataset["Pred_Prob"] = pred
    dataset["Target"] = y
    dataset = dataset.sort_values(by = "Pred_Prob" , ascending = False)
    dataset["Decile"] = pd.qcut(dataset["Pred_Prob"],10, labels = False)
    lift_data  =  pd.DataFrame(dataset["Decile"].value_counts())
    lift_data  =  lift_data.sort_index(ascending = False)
    pos_label  =  pd.DataFrame(dataset[["Target","Decile"]][dataset["Target"] == 1].groupby("Decile").count())
    neg_label  =  pd.DataFrame(dataset[["Target","Decile"]][dataset["Target"] == 0].groupby("Decile").count())
    lift_data  = pd.merge(lift_data , pos_label , how = "left" , left_index = True , right_index = True)
    lift_data  = pd.merge(lift_data , neg_label , how = "left" , left_index = True , right_index = True)
    lift_data.columns =[["Events","Pos_Events","Neg_Events"]]
    lift_data = lift_data.fillna(0)
    
    lift_data["Pos_Event%"]     = lift_data["Pos_Events"]/float(lift_data["Pos_Events"].sum())
    lift_data["Neg_Event%"]     = lift_data["Neg_Events"]/float(lift_data["Neg_Events"].sum())
    lift_data["Event%"]         = round(lift_data["Events"]/float(lift_data["Events"].sum()),2)
    lift_data["Cum_Pos_Event"]  = lift_data["Pos_Event%"].cumsum()
    lift_data["Cum_Neg_Event"]  = lift_data["Neg_Event%"].cumsum()
    lift_data["Cum_Event"]      = lift_data["Event%"].cumsum()
    lift_data["Cumm_Lift"]      = lift_data["Cum_Pos_Event"].values/lift_data["Cum_Event"].values
    lift_data["KS_Stats"]       = lift_data["Cum_Pos_Event"].values - lift_data["Cum_Neg_Event"].values
    
    return lift_data

def xbg_model(params,d_train,d_test ,num_rounds):

    watchlist  = [(d_test,"test"),(d_train,"train")]
    xgb_model = xgb.train(params , d_train, num_rounds, watchlist )
    feature_importance =  pd.DataFrame(list(xgb_model.get_fscore().values()) , index = list(xgb_model.get_fscore().keys()) ,columns= ["Score"])
    feature_importance = feature_importance.sort_values(by ="Score", ascending= False)
    feature_importance.plot(kind  = "barh")
    plt.show()
    y_train_predict = xgb_model.predict(d_train)
    train_auc_score = auc_score(y_train,y_train_predict)
    print("Train_AUC_Score",train_auc_score)
    model_metrics_train = prob_score(y_train,y_train_predict)
    train_lift_data = lift_score(x_train,y_train_predict,y_train)
    
    y_test_predict = xgb_model.predict(d_test)
    test_auc_score =  auc_score(y_test,y_test_predict)
    print("Test_AUC_Score",test_auc_score)
    model_metrics_test = prob_score(y_test,y_test_predict)
    test_lift_data = lift_score(x_test,y_test_predict,y_test)
     
    return model_metrics_train,model_metrics_test ,train_lift_data,test_lift_data


params = {
        'learning_rate' : 0.1,
        'n_estimators' : 1000,
        'max_depth' : 4,
        'min_child_weight': 4,
        'gamma' : 2,
        'lambda' : 3,
        'subsample' : 0.8,
        'colsample_bytree' : 0.8,
        'objective' : 'binary:logistic',
        'nthread' :4,
        'silent' : 1,
        'eval_metrics' : 'auc',
        'seed' : 27}
num_rounds = 100

train_labels = d_train.get_label()
ratio = np.float(np.sum(train_labels == 0)/np.sum(train_labels == 1))
params["scale_pos_weight"] = 1
#weights = np.zeros(len(y_train))
#weights[y_train == 0] = 1
#weights[y_train == 1] = 9
#d_train = xgb.DMatrix(x_train, label = y_train, weight = weights)

#model_metrics_train,model_metrics_test ,train_lift_data,test_lift_data = xbg_model(params,d_train,d_test,num_rounds)



##building the modeling using xgb settings & sklearn
def modelfit(alg, x_train, x_test, y_train,y_test, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x_train[predictors], label = y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(x_train[predictors], y_train,eval_metric='auc')
        
    #Predict training set:
#   dtrain_predictions = alg.predict(x_train[predictors])
    dtrain_predprob = alg.predict_proba(x_train[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    train_auc_score         = auc_score(y_train,dtrain_predprob)
    model_metrics_train     = prob_score(y_train,dtrain_predprob)
    train_lift_data         = lift_score(x_train,dtrain_predprob,y_train)
    print ("AUC Score (Train): %f" % train_auc_score)
    
#     Predict on testing data:
#    dtest_predictions = alg.predict(x_test[predictors])
    dtest_predprob = alg.predict_proba(x_test[predictors])[:,1]
    test_auc_score       = auc_score(y_test,dtest_predprob)
    model_metrics_test   = prob_score(y_test,dtest_predprob)
    test_lift_data       = lift_score(x_test,dtest_predprob,y_test)
    print ('AUC Score (Test): %f' % test_auc_score)
                
    feat_imp = pd.DataFrame(alg.feature_importances_)
    feat_imp.index = predictors
    feat_imp.columns = ["Score"]
    feat_imp = feat_imp.sort_values(by = "Score", ascending = False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    return model_metrics_train,train_lift_data,model_metrics_test,test_lift_data,alg
    
xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

#model_metrics_train,train_lift_data,model_metrics_test,test_lift_data,alg = modelfit(xgb1, x_train, x_test, y_train,y_test, predictors)
 


#Grid seach on subsample and max_features
#Choose all predictors except target & IDcols
param_test1 = {
    'max_depth':np.arange(3,10,2),
    'min_child_weight':np.arange(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=5,
                                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
                       param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(x_train,y_train)
