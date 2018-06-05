# -*- coding: utf-8 -*-
"""
Created on Sat May 26 15:55:56 2018

@author: dillo
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import warnings
warnings.filterwarnings('ignore')


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]

start = time.time()

if __name__ == "__main__":
    # NOTE: on posix systems, this *has* to be here and in the
    # `__name__ == "__main__"` clause to run XGBoost in parallel processes
    # using fork, if XGBoost was built with OpenMP support. Otherwise, if you
    # build XGBoost without OpenMP support, you can use fork, which is the
    # default backend for joblib, and omit this.
    try:
        from multiprocessing import set_start_method
    except ImportError:
        raise ImportError("Unable to import multiprocessing.set_start_method."
                          " This example only runs on Python 3.4")
    set_start_method("forkserver")

    import numpy as np
    from prettytable import PrettyTable
    from sklearn.model_selection import GridSearchCV
    from sklearn.datasets import make_regression
    import pandas as pd
    import xgboost as xgb
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    from sklearn.linear_model import RidgeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import make_scorer
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_validate
    import numpy as np
    import time
    from prettytable import PrettyTable
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    from sklearn import metrics
    from sklearn.linear_model import RidgeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.metrics import make_scorer
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_validate
    from xgboost import XGBClassifier



    rng = np.random.RandomState(7)
    #for scoring xgb with f1 score
    def xgb_f1(y,t):
        t = t.get_label()
        y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y] # binaryzing your output
        return 'f1',metrics.f1_score(t,y_bin)

    ccData = pd.read_csv('creditcard.csv')



    scaler = MinMaxScaler()
    columns = ccData.columns.tolist()
    columns = columns[1:30] # transform V1 through Amount, not time or class
    ccData[columns] =scaler.fit_transform(ccData[columns])


    X = ccData[columns].values
    y = ccData['Class'].values
    
    #X, y = make_classification(n_samples = 10000)
    X= pd.DataFrame(X)
    y = pd.DataFrame(y)

    numCores = 96
    os.environ["OMP_NUM_THREADS"] = str(numCores)
    

    
    numFolds = 3
    i = 7
    inner_cv = StratifiedKFold(n_splits = numFolds, shuffle = True, random_state = i)
    outer_cv = StratifiedKFold(n_splits = numFolds, shuffle = True, random_state = i+10)


    # give shorthand names to models and use those as dictionary keys mapping
    # to models and parameter grids for that model
    logit_model = LogisticRegression()
    xgb_model = XGBClassifier(random_state = 7)
    rf_model = RandomForestClassifier(random_state = 7)
    ridge_model = RidgeClassifier(random_state = 7)
    nb_model = GaussianNB()
    knn_model = KNeighborsClassifier()
    
    models_and_parameters = {'1_Logit': (LogisticRegression(),{'C': [0.01,0.03,0.1,0.3,1,3,10,30]}),
                             '2_RF': (RandomForestClassifier(random_state = 7, n_jobs = numCores), {'n_estimators': [5,10,50,100],'max_features': [1,3,10,'auto',None]}),        
                             '4_NB': (GaussianNB(), {'priors': [[.1,.9],[.2,.8],[.3,.7],[.4,.6],[.5,.5],[.6,.4],[.7,.3],[.8,.2],[.9,.1]]}),
                             '5_KNN': (KNeighborsClassifier(n_jobs = numCores), {'n_neighbors': [1,2,3,4], 'weights': ['uniform','distance']}),
                             '6_XGB': (XGBClassifier(n_jobs = numCores, random_state = 7), {'max_depth': [3,5,10,50,100,200], 'n_estimators': [5,10,50,100,150], 'reg_alpha': [0,0.01,0.1,1,10,100],'reg_lambda': [0,0.01,0.1,1,10,100]})
                             }

    # we will collect the average of the scores on the 3 outer folds in this dictionary
    # with keys given by the names of the models in `models_and_parameters`
    average_scores_across_outer_folds_for_each_model = dict()

    # find the model with the best generalization error
    for name, (model, params) in models_and_parameters.items():
        # this object is a regressor that also happens to choose
        # its hyperparameters automatically using `inner_cv`
        regressor_that_optimizes_its_hyperparams = GridSearchCV(
                estimator=model, param_grid=params,
                cv=inner_cv, scoring='f1', n_jobs = numCores,verbose=1)
        
        # estimate generalization error on the 3-fold splits of the data


        # estimate generalization error on the 3-fold splits of the data
        scoring = {'tp' : make_scorer(tp), 'tn' : make_scorer(tn),
               'fp' : make_scorer(fp), 'fn' : make_scorer(fn), 'f1': 'f1'}
        cv_results = cross_validate(regressor_that_optimizes_its_hyperparams, X, y.values.ravel(), scoring=scoring,return_train_score = False,
                                    cv = outer_cv,verbose=1)
        scores_across_outer_folds = cv_results['test_f1']
        
        for n in range(numFolds):
            t = PrettyTable(['   ', '0','1'])
            t.add_row(['0', cv_results['test_tn'][n],cv_results['test_fp'][n]])
            t.add_row(['1', cv_results['test_fn'][n],cv_results['test_tp'][n]])
            print(t)
		
        #print best parameters for each model
        print(regressor_that_optimizes_its_hyperparams.fit(X,y.values.ravel()).best_estimator_)
        # get the mean MSE across each of outer_cv's 3 folds
        average_scores_across_outer_folds_for_each_model[name] = np.mean(scores_across_outer_folds)
        error_summary = 'Model: {name}\nF1 score in the 5 outer folds: {scores}.\nAverage F1 score: {avg}'
        print(error_summary.format(
            name=name, scores=scores_across_outer_folds,
            avg=np.mean(scores_across_outer_folds)))
        print()

    print('Average score across the outer folds: ',
          average_scores_across_outer_folds_for_each_model)

    many_stars = '\n' + '*' * 100 + '\n'
    print(many_stars + 'Now we choose the best model and refit on the whole dataset' + many_stars)

    best_model_name, best_model_avg_score = max(
            average_scores_across_outer_folds_for_each_model.items(),
            key=(lambda name_averagescore: name_averagescore[1]))

    # get the best model and its associated parameter grid
    best_model, best_model_params = models_and_parameters[best_model_name]

    # now we refit this best model on the whole dataset so that we can start
    # making predictions on other data, and now we have a reliable estimate of
    # this model's generalization error and we are confident this is the best model
    # among the ones we have tried
    final_regressor = GridSearchCV(best_model, best_model_params, cv=inner_cv, n_jobs = numCores,verbose=1)
    final_regressor.fit(X, y.values.ravel())

    print('Best model: \n\t{}'.format(best_model))
    print('Estimation of its generalization error (F1 score):\n\t{}'.format(
		best_model_avg_score))
    print('Best parameter choice for this model: \n\t{params}'
		  '\n(according to cross-validation `{cv}` on the whole dataset).'.format(
		  params=final_regressor.best_params_, cv=inner_cv))

    print ("Process time: ", + (time.time() - start))


