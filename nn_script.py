#!/usr/bin/env python


''' Script wil perfom and CV shiperparameter search to find optimal
        neural network hyperparameters
        using all the combinations specified in 

        DATASETS_TO_USE of datasets
        and 
        LOSSES_TO_USE of nn losses

'''

import os

#CCC (before importing tf)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #2 will print errors,3 not
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



import sys
import getopt
import time
import warnings
import numpy as np
import pandas as pd
import random as random
import joblib
from tabnanny import verbose
import matplotlib.pyplot as plt
import sys
import datetime
import copy

from sklearn.datasets import load_boston
from sklearn.datasets import load_svmlight_file
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  mean_absolute_error
from sklearn.model_selection import  cross_val_predict, cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.utils import losses_utils
from keras.losses import LossFunctionWrapper





######################################################################
##                              MACROS
######################################################################


#   SAVE PATHS
SAVE_PATH =  os.path.dirname('/gaa/home/adrianrp/tfm/')
MODEL_SAVING_NAME = 'mlp_search.nnsearch'
REPORT_SAVING_NAME = ''
DATASETS_PATH =  os.path.join(SAVE_PATH, 'DATASETS/') 



# HYPERPARAMETERS MACROS
N_FOLDS = 5 #at least 2
DEFAULT_EPOCHS = 5000
DEFAULT_BATCH_SIZE = 200
N_BINS = 31 #Error distribution histogram


#   Alpha(regularization) value range
L_ALPHA = [10.**k for k in range(-6, 6)]


#   Huber and PseudoHuber Loss --> Delta value range
l_delta = [1, 1.2, 1.4, 1.6 , 1.8 , 2]
ADDITIONAL_PARAM_GRID_HUBER = {'regressor__mlp__loss__delta': l_delta}  

#   e-insensitive Loss --> epsilon  value range
l_epsilon =  [0, 0.5]
l_epsilon.extend([10.**k for k in range(-4,0)])

ADDITIONAL_PARAM_GRID_E_INSENSITIVE =  {'regressor__mlp__loss__epsilon': l_epsilon} 


#   quantiles for Quantiles Loss --> This wont be searched by CV
# each value of l_quantile will be treated as an indepent loss
#l_cuantile =  [0.05, 0.5, 0.95]
l_cuantile =  [ 0.5]
ADDITIONAL_PARAM_GRID_QUANTILE =  {'regressor__mlp__loss__cuantile': l_cuantile} 


#   ALL DATASETS AND LOSSES AVAIBLE AT SCRIPT (Dont touch this macro unless new ONES its added!)

CLIP_PREDICTIONS = True

DATASETS_TO_USE = [     'boston_housing',
                        'abalone' ,
                        'cpusmall',
                        'space_ga' ,
                        'mg',
                        'auto_mpg'
                ]

DATASETS_TARGET_NAMES = {   'boston_housing' : 'MEDV',
                            'abalone' : 'Rings',
                            'cpusmall' : 'usr',
                            'space_ga' : 'Proporción_Votos_Emitidos',
                            'mg' : 'Mackey-Glass',
                            'auto_mpg': 'mpg'
                        }


# Must specify min and max value
DATASETS_CLIP_RANGE = {     'boston_housing' : (5,50),
                            'abalone' : (0,30),
                            'cpusmall' : (0,100),
                            'space_ga' : (-np.inf, np.inf) ,
                            'mg' : (-np.inf, np.inf),
                            'auto_mpg': (-np.inf, np.inf),
                        }




                 
LOSSES_TO_USE = [       'mse',
                        'mean_absolute_error',
                        'cosine_similarity',
                        'log_cosh',
                        'huber',
                        'pseudohuber', 
                        'e_insensitive',
                        'quantile',
]


TRAIN_SEED = 123
VALIDATION_SEEDS = [111, 222, 333, 444, 555,666] #last one for cross_val_score(), the other for cross_val_predict()




# CCC - SLURM CONFIG
tf.config.set_visible_devices([], 'GPU')
sys.path.append(os.getcwd()) 






######################################################################
##                          CUSTOM LOSSES
######################################################################



###----------------------Epsilon-Insensitive--------------------------

def e_insensitive( y_true, y_pred, epsilon = 1.):
    '''Computes epsilon-insensitive loss between `y_true` and `y_pred` 
    '''

    y_pred = tf.cast(y_pred, dtype=K.floatx())
    y_true = tf.cast(y_true, dtype=K.floatx())
    epsilon = tf.cast(epsilon, dtype=K.floatx())
    
    abs_difference = tf.abs(tf.subtract(y_pred, y_true))
    
    return tf.where(abs_difference > epsilon, abs_difference - epsilon , 0)



class EpsilonInsensitiveLoss(LossFunctionWrapper):
    '''Computes epsilon-insensitive loss between `y_true` and `y_pred`
    
    
    
        Standalone usage:
        y_true = [[0, 1], [0, 0]]
        y_pred = [[0.6, 0.4], [0.4, 0.6]]
        # Using 'auto'/'sum_over_batch_size' reduction type.
        h = EpsilonInsensitiveLoss(epsilon = 1.)
        h(y_true, y_pred).numpy()
        >>0.0
        
        
        # Using 'sum' reduction type.
        y_true = [[0, 1], [0, 0]]
        y_pred = [[0.6, 0.4], [0.4, 0.6]]
        h = EpsilonInsensitiveLoss(epsilon = 0.05,reduction = reduction=tf.keras.losses.Reduction.SUM)
        h(y_true, y_pred).numpy()
        >>1.0

        # Using 'None' reduction type.
        y_true = [[0, 1], [0, 0]]
        y_pred = [[0.6, 0.4], [0.4, 0.6]]
        h = EpsilonInsensitiveLoss(epsilon = 0.05,reduction = tf.keras.losses.Reduction.NONE)
        h(y_true, y_pred).numpy()
        >> array([0.55, 0.45], dtype=float32)


    '''
    
    def __init__(self, 
                 epsilon=1.,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name = 'e_insensitive',
                ):
        
        super(EpsilonInsensitiveLoss, self).__init__(e_insensitive,
                                                     epsilon=epsilon,
                                                     name = name,
                                                     reduction=reduction
                                                    )

    def __repr__ (self):
        return 'e_insensitive'



###----------------------PseudoHuber------------------------------

def pseudohuber( y_true, y_pred, delta = 1.):
    
    assert delta != 0., 'Error, delta cannot be 0, division by 0 not allowed'
    
    y_pred = tf.cast(y_pred, dtype=K.floatx())
    y_true = tf.cast(y_true, dtype=K.floatx())
    delta = tf.cast(delta, dtype=K.floatx()) 
    
    num = tf.math.square( tf.abs( tf.subtract(y_pred, y_true) ) )
    delta_squared = tf.math.square( delta )
    second_op = tf.divide(num,delta_squared) 
    return delta * tf.math.sqrt( tf.convert_to_tensor(1, dtype=second_op.dtype) + second_op )


class PseudohuberLoss(LossFunctionWrapper):
    '''Computes epsilon-insensitive loss between `y_true` and `y_pred`
    

    '''
    
    def __init__(self, 
                 delta=1.,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name = 'pseudohuber',
                ):
        
        super(PseudohuberLoss, self).__init__(  pseudohuber,
                                                delta=delta,
                                                name = name,
                                                reduction=reduction
                                                )
    def __repr__ (self):
        return 'pseudohuber'



###----------------------Quantile------------------------------------

def pinball( y_true, y_pred, cuantile = 0.5):
    
    assert (not cuantile < 0.) or (not cuantile > 1.), 'Error, cuantile must be a value between 0 and 1.'
    
    y_pred = tf.cast(y_pred, dtype=K.floatx())
    y_true = tf.cast(y_true, dtype=K.floatx())
    cuantile = tf.cast(cuantile, dtype=K.floatx()) 
    
    diff = tf.subtract(y_pred, y_true) 
    zeros = tf.zeros_like(diff)  
    
    return (1-cuantile) * tf.math.maximum(-diff,zeros )   + cuantile *  tf.math.maximum(zeros,diff )


class QuantileLoss(LossFunctionWrapper):
    '''Computes epsilon-insensitive loss between `y_true` and `y_pred`
    

    '''
    
    def __init__(self, 
                 cuantile=0.5,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name = 'pinball',
                ):
        
        super(QuantileLoss, self).__init__(     pinball,
                                                cuantile=cuantile,
                                                name = name,
                                                reduction=reduction
                                                )

    def __repr__ (self):
        return 'quantile'
                                                     
        


######################################################################
##                  DATASET_LOADING_LIB
######################################################################



def get_bostonhousing_dataset(source='scikit', debug= False ):
    '''Returns x,y, 'target_name'
    '''
    '''
    vars_housing   = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', \
                          'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT']
    target_housing = ['MEDV']
    '''



    with warnings.catch_warnings():
        # You should probably not use this dataset.
        warnings.filterwarnings("ignore")
        boston = load_boston()

    x = boston['data']
    y = boston['target']

    return x,y, DATASETS_TARGET_NAMES['boston_housing']




#loads dataset from libsvm file
def get_data_from_libsvmfile(file_path):
    data = load_svmlight_file(file_path)
    return data[0].toarray(), data[1]



def get_dataset(dataset_name,dataset_path = '', debug = False):


    if(dataset_name == 'boston_housing' ):
        return  get_bostonhousing_dataset(source='scikit', debug= debug )
    else:
        file_name = dataset_name + '.libsvm'


        if(dataset_path):
            file_path = os.path.join(dataset_path, file_name)
        else:
            file_path = file_name

        x, y  = get_data_from_libsvmfile(file_path)


        return  x,y, DATASETS_TARGET_NAMES[dataset_name]                    









######################################################################
##                  TRAIN_SCRIPT
######################################################################

def keras_fnn_builder(  n_features, 
                        hidden_layers_sizes, 
                        #optimizer, 
                        #loss, 
                        #metrics,
                        alpha=0.0001
                    ):
    
    '''
        n_features: number of input features
    '''
    
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(n_features,)))
    
    for hidden_layer_size in hidden_layers_sizes:
        model.add(keras.layers.Dense(hidden_layer_size, activation="relu",  kernel_regularizer=keras.regularizers.l2(l=alpha)))
        
    model.add(keras.layers.Dense(1,activation = 'linear',name='output'))
    
    #Compile the model before returning it
    #model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model





# Returns final wrapped model
def create_scikerasmodel(   n_features = -1,
                            hidden_layers_sizes = (20, 20),
                            loss='mse',
                            optimizer='adam',  
                            metrics=['mae'],
                            alpha = 1.e-2,
                            batch_size = DEFAULT_BATCH_SIZE,
     
                            use_tolerance = False,
                            tolerance = 1.e-3,
                            patience = 10,  
                            epochs = 1500,  
            
                            seed = TRAIN_SEED,
                            verbose = 0,

                        ):

    ### WRAPPER CREATION

    # Tolerance for loss, DO NOT CONFUSE WITH EARLY STOPPING (stop if after N epochs VALIDATION ERROR do not drecreases a min. specified by patience)
    if(use_tolerance):
        callbacks = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta = tolerance,patience = patience, verbose = verbose)
    else:
        callbacks = []
    

    mlp = KerasRegressor(
 
        #first keras function builder params
        model=keras_fnn_builder,
        n_features = n_features, 
        hidden_layers_sizes = hidden_layers_sizes,
        optimizer=optimizer, 
        loss=loss, 
        metrics=metrics,
        alpha = alpha,
        verbose = verbose, 
        
        #Then SciKeras params
        callbacks = callbacks ,
        random_state = seed,
        warm_start = False,
        batch_size = batch_size,
        epochs = epochs,
        )
        

    # Creation of a Pipeline
    regr = Pipeline(steps=[('std_sc', StandardScaler()),
                           ('mlp', mlp)])

    # Using a metaestimator to escalate also the target
    y_transformer = StandardScaler()
    inner_estimator = TransformedTargetRegressor(regressor=regr,
                                                 transformer=y_transformer)

    return inner_estimator





def train_nn(   x,
                y,
                saving_name='mlp_results.nnsearch',
                saving_route= '',
                hidden_layers_sizes = (20, 20),
                loss='mse',
                optimizer='adam',  
                metrics=['mae'],

                n_folds = 5,
                alpha = 1.e-2,
                l_alpha = [10.**k for k in range(-6, 6)],
                additional_hp_grid = None,
                cv_scoring_metric = 'neg_mean_absolute_error',

                use_tolerance = False,
                tolerance = 1.e-3,
                patience = 10,  
                epochs = 1500,  
                batch_size = DEFAULT_BATCH_SIZE,

                seed = TRAIN_SEED,
                n_jobs = None,
                compress_level = 3,
                verbose = 0,
                debug = False

            ):

    '''Trains nn

        Args:
            hidden_layers_sizes(tuple): #length = n_layers - 2, The ith element represents the number of neurons in the ith hidden layer.
            saving_name=name given to the search info
            saving_route= path where search info will be saved, if '' current directory will be choosen.
            loss:   'mse',
                    'cosine_similarity',
                    'mean_absolute_error'
                    tf.keras.losses.Huber      ----> Since it need hyperparameter,delta!
                    'log_cosh'
                    'cosine_similarity'

            alpha: regularization hyperparameter
            l_alpha: regularization hyperparameter values
            additional_hp_grid: grid of additional hipèrparameters to include during CV search
                    CAUTION! if the loss has and hipperparameter,it should be called like this, using keras huber loss as example:
                    param_grid = {'regressor__mlp__loss__delta': l_delta}  


            use_tolerance: whether to stop training if training error not decreas at least tolerance after each epoch of training. DO NOT CONFUSE with 
                            early stopping (which means if the training stops if each after N number or epochs, VALIDATION error do not decreases, used
                            in order to avoid overfitting)
            tolerance: Minimum change in training loss to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
            epochs: MAXIMUM n of epochs the model will be trained if use_tolerance condition is not met

            seed(None or int): Number or None 
            n_jobs(int):
            compress_level: int from 0 to 9 or bool or 2-tuple.  Higher value means more compression,
                            but also slower read and write times. Using a value of 3 is often a good compromise. 
            verbose: if not set to 0, will show verbose when calling sklearn sklearn.model_selection.cross_val_predict func.
            ...
    '''

    # Initial checks... 

    if(seed): 

        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED']=str(seed)
        '''
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
        '''
    
    ### WRAPPER CREATION
    inner_estimator = create_scikerasmodel(     n_features =  x.shape[1],
                                                hidden_layers_sizes = hidden_layers_sizes,
                                                loss= loss,
                                                optimizer= optimizer,  
                                                metrics= metrics,
                                                alpha = alpha,
                                                batch_size = batch_size,
                                     
                                                use_tolerance = use_tolerance,
                                                tolerance = tolerance,
                                                patience = patience,  
                                                epochs = epochs,  

                                                seed = seed,
                                                verbose = verbose,
                                            )

    ### CV SEARCH CONFIG

    param_grid = {'regressor__mlp__alpha': l_alpha} 

    if(additional_hp_grid is not None):
        param_grid.update(additional_hp_grid)

    kf = KFold(n_folds, shuffle=True, random_state = seed)

    cv_estimator = GridSearchCV(inner_estimator, 
                                param_grid=param_grid, 
                                cv=kf, 
                                scoring=cv_scoring_metric,
                                return_train_score=True,
                                refit=True,
                                n_jobs=n_jobs, 
                                verbose=verbose)



    ### TRAINING TIME

    if(verbose):
        print('Training with CV Search...')
    t_0 = time.time()
    cv_estimator.fit(x, y)
    t_1 = time.time() 
    cv_search_time =  (t_1 - t_0)/60.
    if(verbose):
        print('CV Search Completed in {cv_search_time:.2f}  minutes! \n Saving results....')


    ### SAVING RESULTS

    #Additional training info
    #       hp values range will be saved as 'hp_name': [values range]
    saving_info =  {    'cv_estimator' : cv_estimator,
                        'cv_search_time': cv_search_time,
                        'cv_seach_n_jobs':n_jobs,
                        'seed': seed,
                        'l_alpha': l_alpha,
                        'additional_hp': [],
                        'epochs': epochs,
                        'use_tolerance' :use_tolerance,
                        'tolerance':tolerance, 
                        'patience':patience,
                        'batch_size': batch_size,
                    }
    if(additional_hp_grid is not None):# if there were more hiperparameters in the cv search we store its info.
        saving_info['are_additional_hp'] = True
        for k,v in additional_hp_grid.items(): #TODO Paralelizar esta operacion aunque tampoco es muy importante hacerlo
            saving_info['additional_hp'].append(k)
            saving_info[k] = v

    

    if(saving_route):
        os.makedirs(saving_route, exist_ok=True)
        complete_path = os.path.join(saving_route, saving_name )
    else:
        complete_path =  saving_name 
        saving_route = os.getcwd()



    with open(complete_path, 'wb') as f:  
        saving_files = joblib.dump(value = saving_info, filename= f, compress=compress_level)













######################################################################
##                  AUTOREPORT_SCRIPT
######################################################################


def load_model( filename,  
                verbose = 0):

    #Additional training info
    saving_info = joblib.load(filename)

    if(verbose):
        print('Done!')

    return saving_info


def update_log_status( debug= True, report_folder_name = '', message = '' ):
    '''if debug prints execution evolution messages passes onto report folder name'''
    if(debug):
        file = open(os.path.join(report_folder_name, 'log_execution_evolution.txt'), 'a+')
        print( str(datetime.datetime.now()) + '\t\t\t' + message + '\n',file=file)
        file.close()





def plot_preddispersion_and_errdistribution(y, cv_y_pred_mean, target_name,save_path,save_report,bins,scores_005 = None , scores_095 = None, clipped = False):   
    
    # DIAGRAMA DE DISPERSION DE LAS PREDICCIONES
    plt.title('Errores MAE: Reales vs. Predichos')
    plt.xlabel(target_name)
    plt.ylabel(target_name + '_predichos')
    _ = plt.plot(y, cv_y_pred_mean, '.', y, y, '-')
    if((scores_005 is not None) and (scores_095 is not None)):
        plt.fill_between(y.ravel(), scores_005.ravel(), scores_095.ravel(), color='b', alpha=.1)

    if(save_report):
        fig_name = 'Real_vs_Predicted_MAE_Values'
        if(clipped):
            fig_name = fig_name + '_clipped'
        plt.savefig(os.path.join(save_path,fig_name))
    else:
        plt.show()

    plt.close()


    # HISTOGRAMA DE DISPERSION DE LOS ERRORES
    err = y - cv_y_pred_mean
    plt.title("Validación cruzada: Histograma de Errores")
    plt.xlabel("Errores")
    plt.ylabel("Frecuencia de los Errores")
    _ = plt.hist(err, bins=bins)
    if(save_report):
        fig_name = 'MAE_ERROR_DISTRIBUTION'
        if(clipped):
            fig_name = fig_name + '_clipped'
        plt.savefig(os.path.join(save_path,fig_name))
    else:
        plt.show()

    plt.close()






def plot_CV_model_evaluation(   x,
                                y, 
                                target_name, 
                                model, 
                                n_folds,
                                bins= 31, 
                                n_cross_val_predict_execs = 5,
                                n_cross_val_predict_seeds = VALIDATION_SEEDS,
                                n_jobs = None,
                                history = None,
                                save_report = False,
                                dataset_name = '',
                                save_path= 'save_path',
                                file = sys.stdout,
                                debug = True
                            ):

    '''
    
        Args:
            file: output to write report info
    '''

    #cross_val_predict()
    print('EXECUTING CROSS_VAL_PREDICT() ' + str(n_cross_val_predict_execs) + ' times...', file = file)
    cv_predict_mae = None
    cv_predict_maes = []
    l_cv_y_pred = []
    l_cv_y_pred_clipped = []


    if(n_cross_val_predict_seeds):
        random_states = n_cross_val_predict_seeds
    else:
        random_states = [None for i in range (n_cross_val_predict_execs)]


    if(CLIP_PREDICTIONS):
        min_clip = DATASETS_CLIP_RANGE[dataset_name][0]
        max_clip= DATASETS_CLIP_RANGE[dataset_name][1]

    t_0 = time.time()
    for i in range(n_cross_val_predict_execs):
    
        kf = KFold(n_folds, shuffle=True, random_state= random_states[i])
        # Model weights are resetting in each fold for cross_val_predict
        cv_y_pred = cross_val_predict(model, x, y.ravel(), cv=kf, verbose = 0, n_jobs = n_jobs) 
        l_cv_y_pred.append(cv_y_pred)

        if(CLIP_PREDICTIONS):
            l_cv_y_pred_clipped.append(np.clip(cv_y_pred, min_clip,max_clip))

        # MAE per single execution
        cv_predict_mae = mean_absolute_error(y, cv_y_pred)
        cv_predict_maes.append(cv_predict_mae)



    t_1 = time.time() 
    time_spent =  (t_1 - t_0)/60.
    print(f'\t\t completed in {time_spent:4f} minutes.\n', file = file)
    
    #Model Predictions
    #       For each pattern, we calculate its prediction based on the mean 
    #       of the n_cross_val_predict_execs cross_val_predict() executions.
    #       Then score metric will be calculated with this
    a_cv_y_pred = np.array(l_cv_y_pred).T
    cv_y_pred_mean = a_cv_y_pred.mean(axis=1)
    cv_y_pred_median = np.median(a_cv_y_pred, axis=1)


    #Model CLIPPED Predictions
    if(CLIP_PREDICTIONS):
        a_cv_y_pred_clipped = np.array(l_cv_y_pred_clipped).T
        cv_y_pred_mean_clipped = a_cv_y_pred_clipped.mean(axis=1)
        cv_y_pred_median_clipped = np.median(a_cv_y_pred_clipped, axis=1)

    # Model Scores
    model_mean = mean_absolute_error(y.ravel() , cv_y_pred_mean)
    model_median = mean_absolute_error(y.ravel(), cv_y_pred_median)

    if(CLIP_PREDICTIONS):
        model_mean_clipped = mean_absolute_error(y.ravel() , cv_y_pred_mean_clipped)
        model_median_clipped = mean_absolute_error(y.ravel(), cv_y_pred_median_clipped)


    print("MODEL SCORE:" ,file = file)
    print("\t\t ---> MEAN MAE: %.10f" % model_mean , file = file)
    print("\t\t ---> MEDIAN MAE: %.10f" % model_median , file = file)
    print("\n\n\n" , file = file)

    if(CLIP_PREDICTIONS):
        print("MODEL SCORE OF CLIPPED PREDICTIONS:" ,file = file)
        print("\t Clipping Interval (min, max):   (" + str(min_clip)  + ' , '+ str(max_clip) + ')' , file = file)
        print("\t\t ---> MEAN MAE: %.10f" % model_mean_clipped , file = file)
        print("\t\t ---> MEDIAN MAE: %.10f" % model_median_clipped , file = file)
        print("\n\n\n" , file = file)



    
    print("**************************************************************************" ,file = file)
    print("NOTE DONT USE THIS SCORE AS RESULT, ONLY AS A REFERENCE FOR DEBUGING" ,file = file)
    print("**************************************************************************" ,file = file)
    print("MODEL SCORES per single cross_val_predict() CALL:" ,file = file)
    print("\t\t ---> Predict MAE CV Errors:\n", file = file)
    print(cv_predict_maes, file = file)
    print("\t\t ---> Predict MAE CV Errors MEAN: %.10f" % np.mean(cv_predict_maes) , file = file)
    print("\t\t ---> Predict MAE CV Errors STD: %.10f" % np.std(cv_predict_maes) , file = file)
    print("\t\t ---> Predict MAE CV Errors MEDIAN: %.10f" % np.median(cv_predict_maes)  , file = file)
    print("**************************************************************************\n\n\n\n" ,file = file)



    print('EXECUTING CROSS_VAL_SCORE() ...', file = file)
    kf = KFold(n_folds, shuffle=True, random_state= random_states[-1]) #seed = 666
    t_0 = time.time() 
    cv_val_scores  = cross_val_score(model, x, y.ravel(), scoring="neg_mean_absolute_error", cv=kf, n_jobs=n_jobs, verbose = 0)
    t_1 = time.time() 
    time_spent =  (t_1 - t_0)/60.
    print(f'\t\t completed in {time_spent:4f} minutes.\n', file = file)
    print("\t\t ---> MAE Mean: %.10f" % -cv_val_scores.mean() , file = file)
    print("\t\t ---> MAE Std: %.10f" % cv_val_scores.std() , file = file)
    print("\n\t\t ---> MAE Scores: \n",  file = file)
    print(-np.round(cv_val_scores, 4), file= file)
    print('\n\n', file = file)


    #print l_cv_y_pred,used for calc. cv_y_pred_mean and cv_y_pred_median to check results
    if(debug):
        print('\n\n\n\n' , file = file)
        print('-------------------------------------------------------------' , file = file)
        print('DEBUG INFO', file = file)
        print('-------------------------------------------------------------', file = file)
        print("\n All predictions generated of all cross_val_predict_executions (l_cv_y_pred): \n" ,file = file)
        print(l_cv_y_pred ,file = file)




    print('\n\n\n GENERATING PLOTS WITH  MEAN PREDICTIONS!', file = file)

    if('Quantile' in str(model.regressor_.named_steps['mlp'].loss) ):
        print('\t\t\t QUANTIL LOSS!', file = file)
        model_005 = copy.deepcopy(model)
        model_095 = copy.deepcopy(model)
        model_005.regressor_.named_steps['mlp'].loss = QuantileLoss(cuantile = 0.05)
        model_095.regressor_.named_steps['mlp'].loss = QuantileLoss(cuantile = 0.95)
        print('\t\t\t\t cross_val_score for 0.05 and 0.95 quantiles!', file = file)

        t_0 = time.time() 
        scores_005  = cross_val_predict(model_005, x, y.ravel(), cv=kf, n_jobs=n_jobs, verbose = 0)
        scores_095  = cross_val_predict(model_095, x, y.ravel(), cv=kf, n_jobs=n_jobs, verbose = 0)

        t_1 = time.time() 
        time_spent =  (t_1 - t_0)/60.
        print(f'\t\t\t\t\t completed in {time_spent:4f} minutes.\n', file = file)

        plot_preddispersion_and_errdistribution(y, cv_y_pred_mean, target_name,save_path,save_report,bins,
                                                scores_005 = scores_005 , scores_095 = scores_095,
                                                clipped = False)  


        if(debug):
            print('\n\n\n\n' , file = file)
            print('-------------------------------------------------------------' , file = file)
            print('DEBUG INFO', file = file)
            print('-------------------------------------------------------------', file = file)
            print("\n Final predictions generated for median with Quantile Regression: (cv_y_pred_mean.ravel()): \n" ,file = file)
            print(cv_y_pred_mean.ravel() ,file = file)
            print('\t\t-------------------------------------------', file = file)
            print("\n TRUE TARGETS : (y.ravel()): \n" ,file = file)
            print(y.ravel() ,file = file)
            print('\t\t-------------------------------------------', file = file)
            print("\n Predictions  for Quantile=0.05: (scores_005.ravel()): \n" ,file = file)
            print(scores_005.ravel() ,file = file)
            print('\t\t-------------------------------------------', file = file)
            print("\n Predictions  for Quantile=0.95: (scores_095.ravel()): \n" ,file = file)
            print(scores_095.ravel() ,file = file)



    else:

        plot_preddispersion_and_errdistribution(y, cv_y_pred_mean, target_name,save_path,save_report,bins,
                                                scores_005 = None , scores_095 = None,
                                                clipped = False)  
        plot_preddispersion_and_errdistribution(y, cv_y_pred_mean_clipped, target_name,save_path,save_report,bins,
                                                scores_005 = None , scores_095 = None,
                                                clipped = True)  


    
    print('\n\n\n\n' , file = file)
    print('-------------------------------------------------------------' , file = file)
    print('Loss Evolution', file = file)
    print('-------------------------------------------------------------', file = file)
    print('\t\t\t ' + str(len(history['loss'])) + ' epochs used to complete training.', file = file)

    
    print('\n\n\n\n---> Trainin MAE Error improvement per epoch:', file = file)
    training_error_improvement = [a - b for a,b in zip(history['loss'] , history['loss'][1:]) ] 
    print(training_error_improvement, file = file)


    print('\n\n\n\n---> Training MAE Error value per epoch', file = file)
    print(history['loss'], file = file)





    # summarize history for loss
    if(history):
        _ = plt.plot(history['loss'])
        plt.title('Progreso del Error del Mejor Modelo')
        plt.ylabel('Función de Pérdida')
        plt.xlabel('Época')
        if(save_report):
            fig_name = 'Best_Model_Loss_Progress'
            plt.savefig(os.path.join(save_path,fig_name))
        else:
            plt.show()

        plt.close()







#Evolución del la busqueda de un hiperparámetro
def plot_cv_hp_search_report(   saving_info,
                                save_report = False, 
                                save_path= '',
                                file = sys.stdout
                                ):
    '''hp hiperparameter
        Args:

    '''

    cv_estimator = saving_info['cv_estimator']
    n_of_epcohs = len(cv_estimator.best_estimator_.regressor_['mlp'].history_['loss'] )


    print('-------------------------------------------------------------', file = file)
    print('HYPERPARAMETERS SEARCH RESULTS', file = file)
    print('-------------------------------------------------------------', file = file)
    print('\n', file = file)
    print('Best MAE model(CV Search) = %.10f' % (-cv_estimator.best_score_), file = file)
    print('\t trained with  {0:d}'.format(n_of_epcohs) + ' epochs.\n', file = file)


    #Print alpha info
    hp_name = 'alpha'
    hp_values_range = saving_info['l_alpha']
    print('---> HYPERPARAMETER : ' + hp_name, file = file)
    print(hp_name + ' Values Range: ' + str((np.array(hp_values_range).min())) + ' - ' + str(np.array(hp_values_range).max())  , file = file)
    print('Values:', file = file)
    print(hp_values_range,file = file)
    print('\nBest '+ hp_name + ' Value  = ' + str((cv_estimator.best_params_['regressor__mlp__alpha'])) , file = file)

    if( not saving_info['additional_hp']):
        #Plot only ¡ ALPHA(REgularization hp) evolution

        plt.xticks(range(len(hp_values_range)), hp_values_range, rotation=45)
        _ = plt.plot( -cv_estimator.cv_results_['mean_test_score'])
        plt.title("Validación Cruzada: Resultados de las Búsqueda de Hiperparámetros")
        plt.xlabel(hp_name)
        plt.ylabel('Error MAE')

        
    else: # serveral hiperparameters, plot must be done in a different way

        for hp in saving_info['additional_hp']:

            hp_name = hp.split('_')[-1]
            hp_values_range = saving_info[hp]
            print('\n', file = file)
            print('---> HYPERPARAMETER : ' + hp_name, file = file)
            print(hp_name + ' Values Range: ' + str((np.array(hp_values_range).min())) + ' - ' + str(np.array(hp_values_range).max())  , file = file)
            print('Values:' ,file = file)
            print(hp_values_range,file = file)
            print('\nBest '+ hp_name + ' Value  = ' + str(cv_estimator.best_params_[hp]), file = file)

        #param 2 is alpha
        #param1 is the other one 
        
        # Get Test Scores Mean and std for each grid search
        grid_param_2_name = saving_info['additional_hp'][0]
        grid_param_2 = saving_info[grid_param_2_name]
        grid_param_2_name = grid_param_2_name.split('_')[-1] #delete dict idex info from name
        grid_param_1 = saving_info['l_alpha']
        scores_mean = np.array(cv_estimator.cv_results_['mean_test_score']).reshape((len(grid_param_2), len(grid_param_1)),order = 'F')

        # Plot Grid search scores

        # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
        plt.xscale("log")
        plt.xticks(grid_param_1)
        plt.xticks(rotation=45)


        for idx, val in enumerate(grid_param_2):
            _= plt.plot(grid_param_1, -scores_mean[idx], '-', label= grid_param_2_name + ': ' + str(val))

        plt.title('Validación Cruzada: Búsqueda de Hiperparámetros', fontweight="bold")
        plt.xlabel('alpha')
        plt.ylabel('Error MAE')
        plt.legend(loc="best")

        

    if(save_report):
        fig_name = 'CV_Hyperparameter_Results' 
        plt.savefig(os.path.join(save_path,fig_name), bbox_inches='tight')
    else:
        plt.show()

    plt.close()
    print('*************************************************************\n\n\n\n', file = file)






 

def plot_cv_search_complete_report(     x, 
                                        y, 
                                        saving_info, 
                                        target_name, 
                                        bins = 31, 
                                        n_cross_val_predict_execs = 5,
                                        n_cross_val_predict_seeds = VALIDATION_SEEDS,
                                        n_jobs = None,
                                        save_report= False,
                                        dataset_name = '',
                                        save_path='',
                                        save_name= 'report_folder_name',
                                        debug = False
                                        ):
    '''Basically calls all the above functions
        hp:hiperparameter

        Args:

            saving_info: trainig info dict with cvsearch object included
            save_path: path. Recommendable to indicate a folder name
            save_report(bool): if true will save report and plot images in the a folder

            n_cross_val_predict_execs(int): number of times to execute cross_val_predict()
            n_cross_val_predict_seeds
            n_cross_val_predict_seeds(None or list): list of size n_cross_val_predict_execs, to 
                                            make results reproducible. If None no seed will
                                            be choosen
            save_path: route where the report will be saved if save_report is true
            save_name= reports FINAL FOLDER name if save_report is true
    '''
    assert not save_report or (save_report and save_path) or (save_report and save_name), 'Save report selected. Must indicate a valid path and/or filename.'
    assert n_cross_val_predict_execs >=1 , 'n_cross_val_predict_execs should be 1 or greater'
    assert ( (n_cross_val_predict_seeds is None) or ( len(n_cross_val_predict_seeds) == n_cross_val_predict_execs  or len(n_cross_val_predict_seeds) == n_cross_val_predict_execs+1  )   ), 'n_cross_val_predict_seeds Should have the same length as n_cross_val_predict_execs'


    #Opening file to save report if option choosen
    #things will be saved in SAVE_FOLDER
    if(save_report):
        if(save_path):
            if(save_name):
                save_folder = os.path.join(save_path,save_name)
            else:
                save_folder = save_path
        else:
            save_folder = save_name
        os.makedirs(save_folder, exist_ok=True)
        file = open(os.path.join(save_folder, 'report.txt'), 'w+')
    else:
        save_folder = ''
        file = sys.stdout


    cv_estimator = saving_info['cv_estimator']
    #n_folds = saving_info['N_FOLDS']
    n_folds = cv_estimator.n_splits_
    model = saving_info['cv_estimator'].best_estimator_.regressor_.named_steps['mlp']

    epochs = saving_info['epochs']
    use_tolerance = saving_info['use_tolerance']
    tolerance = saving_info['tolerance']
    patience = saving_info['patience']
    batch_size = saving_info['batch_size']

    # CV model info
    print('-------------------------------------------------------------' , file = file)
    print('SEARCH INFO', file = file)
    print('-------------------------------------------------------------', file = file)
    search_time = saving_info['cv_search_time']
    n_jobs = saving_info['cv_seach_n_jobs']
    print(f'CV Search Time:  {search_time:.4f}  min.' , file = file)
    print(f'\t\t using n_jobs= {n_jobs}.', file = file)
    print(f'\t\t and {n_folds} cv folds.', file = file)

    print(f'\t\t TRAINING INFO PER MODEL:', file = file)
    print(f'\t\t\t MAX. Train allowed epochs {epochs}', file = file)
    print(f'\t\t\t Stop train if loss not drecreasing: {use_tolerance}', file = file)
    if(use_tolerance):
        print(f'\t\t\t\t Tolerance: {tolerance}', file = file)
        print(f'\t\t\t\t Patience: {patience}', file = file)
    print(f'\t\t\t Batch Size: {batch_size}', file = file)



    print('\n Best Model Info:', file = file)
    print('\t\t %.4f minutes for refitting the best model.' % (cv_estimator.refit_time_/60.), file = file)
    print(f'\t\t trained with: {model.n_features} features.', file = file)
    print(f'\t\t dataset size: {x.shape[0]}.', file = file)
    print('\t\t hidden layer sizes: ', model.hidden_layers_sizes, file = file)
    print('\t\t training loss: ' + str(model.loss), file = file)
    print('\t\t training seed: ', model.random_state, file = file)
    if(debug):
        print(cv_estimator.best_estimator_.regressor_.named_steps['mlp'], file = file)



    print('*************************************************************\n\n\n\n', file = file)


    # CV hp search results
    plot_cv_hp_search_report(   saving_info,
                                save_report = save_report, 
                                save_path= save_folder,
                                file = file
                                )



    print('-------------------------------------------------------------', file = file)
    print('EVALUATING NOW MODEL WITH BEST MAE SCORE', file = file)
    print('-------------------------------------------------------------', file = file)
    #Best model obtained performance
    plot_CV_model_evaluation(   x, 
                                y, 
                                target_name, 
                                cv_estimator.best_estimator_, 
                                bins= bins, 
                                n_folds = n_folds,
                                n_cross_val_predict_execs = n_cross_val_predict_execs,
                                n_cross_val_predict_seeds = n_cross_val_predict_seeds,
                                n_jobs = n_jobs,
                                history = model.history_,
                                save_report= save_report,
                                dataset_name = dataset_name,
                                save_path= save_folder,
                                file = file,
                                debug = debug
                            )

    


    if(save_report):

        if(True):

            print('\n\n\n\n' , file = file)
            print('-------------------------------------------------------------' , file = file)
            print('DEBUG INFO', file = file)
            print('-------------------------------------------------------------', file = file)
            for k,v in cv_estimator.cv_results_.items():
                if(k == 'params'):
                    print(str(k) + ' : ',file=file)
                    [print(i,file=file) for i in cv_estimator.cv_results_['params']]
                else:
                    print(str(k) + ' : ' + str(v),file=file)
                print('\n\n',file=file)
        file.close()






















######################################################################
##                  META SCRIPT
######################################################################

def main( argv):


        '''
                report_name(string): to make easir various runs over the ccc
                datasets: 'all' or list with names
                losses: 'all' or list with names
                debug: if True prints in terminal 
        '''

        ####################################
        # SCRIPT ARGUMENTS PARSING
        arg_help = '--------------------------------------------------------------------\n'

        arg_help = arg_help +  "{0} \n\t -r,--report_name <str> \n\t -n,--n_jobs <int> \n\t -d,--datasets <'all' | comma separated names> \
                \n\t -l,--losses <'all' | comma separated names> \
                \n\t -v,--debug <bool> \
                \n\t -s,--size comma-separated-ints \
                \n\t -e,--epochs <int>  \
                \n\t -t,--tolerance <int> (without this option training is fixed for --epochs epochs) \
                \n\t -p,--patience <int> (without this option training is fixed for --epochs epochs) \
                \n\t -b,--batch_size <int>".format(argv[0])

        arg_help = arg_help + '\n \n CALL EXAMPLE:\n  {0} --report_name report  --n_jobs 5 --datasets boston_housing --losses mse --size 20,20 --epochs 5000  --tolerance 1.e-16 --patience 500 --batch_size 200 --debug True'.format(argv[0])
        arg_help = arg_help + '\n--------------------------------------------------------------------\n'        
        report_name = None
        n_jobs = None
        datasets = 'all'
        losses = 'all'
        debug = True
        hidden_layers_sizes = None
        epochs = DEFAULT_EPOCHS
        use_tolerance = False
        tolerance =  None 
        patience = None 
        batch_size = DEFAULT_BATCH_SIZE

        try:
            opts, args = getopt.getopt(argv[1:], "r:n:d:l:s:e:t:p:b:v:h", ["report_name=", "n_jobs=", 
                                                                "datasets=", "losses=","size=",
                                                                 "epochs=","tolerance=","patience=", "batch_size=",
                                                                 "debug=", "help"]
                                        )
        except:
            print(arg_help)
            sys.exit(2)
    
        

        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print('\n USAGE:')
                print(arg_help)  # print the help message
                sys.exit(2)
            elif opt in ("-r", "--report_name"):
                report_name = arg
            elif opt in ("-n", "--n_jobs"):
                n_jobs = arg
            elif opt in ("-d", "--datasets"):
                datasets = arg
            elif opt in ("-l", "--losses"):
                losses = arg
            elif opt in ("-s", "--size"):
                hidden_layers_sizes = arg
            elif opt in ("-e", "--epochs"):
                epochs = arg
            elif opt in ("-t", "--tolerance"):
                tolerance = arg
            elif opt in ("-p", "--patience"):
                patience = arg
            elif opt in ("-v", "--debug"):
                debug = arg
            elif opt in ("-b", "--batch_size"):
                batch_size = arg


        #Mandatory options

        if((report_name is None) or (n_jobs is None) or (hidden_layers_sizes is None) ):
            print('ERROR: --report_name , --n_jobs and  --size   options ARE MANDATORY!!')
            print('USAGE:')
            print(arg_help)  # print the help message
            sys.exit(2)


        try:
            n_jobs = int(n_jobs)
            hidden_layers_sizes = tuple([int(i) for i in hidden_layers_sizes.split(',')])
            epochs = int(epochs)
            batch_size = int(batch_size)
            if(tolerance is None or patience is None):
                use_tolerance = False
            else:
                use_tolerance = True
                tolerance = float(tolerance)
                patience = int(patience)
        except:
            print(' ERROR: Argument types are not correct!')
            print('USAGE:')
            print(arg_help)  # print the help message
            sys.exit(2)



        if(   (type(report_name) != str) 
                or (type(n_jobs) != int) 
                or ((type(datasets) != str) ) 
                or (type(losses) != str) 
                or (type(epochs) != int) 
                or (type(batch_size) != int) 
                or ( debug not in ('True','true','False','false')  ) 
            ):

            print('ERROR: Arguments types are not correct!')
            print('\n\n USAGE:')
            sys.exit(2)



        if(datasets != 'all'):
            datasets = [str(i) for i in datasets.split(',')]

        if(losses != 'all'):
            losses = [str(i) for i in losses.split(',')]


        if(debug == 'True' or debug == 'true' ):
            debug = True
        else:
            debug = False


        assert report_name, 'A save path must be specified'
        save_path = os.path.join(SAVE_PATH, report_name )
        os.makedirs(save_path, exist_ok=True)


        if(datasets == 'all'):
                datasets = DATASETS_TO_USE

        if(losses == 'all'):
                losses = LOSSES_TO_USE



        ####################################
        # SCRIPT EXECUTION
        file = open(os.path.join(save_path, 'log_execution_evolution.txt'), 'w')
        file.close()
        script_t_1 = time.time() 
        update_log_status( debug= debug, report_folder_name = save_path, message = 'STARTING REPORT...' )
        update_log_status( debug= debug, report_folder_name = save_path, message = '\t\t hidden layer sizes are: ' + str(hidden_layers_sizes) )
        update_log_status( debug= debug, report_folder_name = save_path, message = '\t\t MAX. Train allowed epochs: ' + str(epochs) )
        update_log_status( debug= debug, report_folder_name = save_path, message = '\t\t Stop Training if loss is not decreasing: ' + str(use_tolerance)  + '\t\t\t (False means models have been trained the maximun number of epochs.)' )
        if(use_tolerance):
            update_log_status( debug= debug, report_folder_name = save_path, message = '\t\t\t Tolerance: ' + str(tolerance) )
            update_log_status( debug= debug, report_folder_name = save_path, message = '\t\t\t Patience: ' + str(patience) )


        for dataset_name_i in datasets:
                ####################################
                #   DATA LOADING
                update_log_status( debug= debug, report_folder_name = save_path, message = '-----> SWITCHING DATASET TO: ' + dataset_name_i )

                x,y, target_name =  get_dataset(       dataset_name = dataset_name_i, 
                                                                dataset_path = DATASETS_PATH, 
                                                                debug = debug
                                                        )

                for loss_i in losses:
                        update_log_status( debug= debug, report_folder_name = save_path, message = '\t\t-----> USING NOW LOSS:' + loss_i )


                        report_folder_name = 'report__loss_' + str(loss_i) + '__dataset_' + str(dataset_name_i) 
                        save_path_i = os.path.join(save_path, report_folder_name )


                        if(loss_i == 'huber'):
                            loss_i = tf.keras.losses.Huber
                            additional_param_grid = ADDITIONAL_PARAM_GRID_HUBER
                        elif(loss_i == 'pseudohuber'):
                            loss_i = PseudohuberLoss
                            additional_param_grid = ADDITIONAL_PARAM_GRID_HUBER
                        elif(loss_i == 'e_insensitive'):
                            loss_i = EpsilonInsensitiveLoss
                            additional_param_grid = ADDITIONAL_PARAM_GRID_E_INSENSITIVE
                        elif( 'quantile' in loss_i):
                            #cuantile = loss_i.split('_')[1]
                            loss_i = QuantileLoss
                            additional_param_grid = None
                        else:
                            additional_param_grid = None

                        ####################################
                        #   TRAINING

                        train_nn(   x,
                                    y,
                                    saving_name= MODEL_SAVING_NAME,
                                    saving_route= save_path_i,
                                    hidden_layers_sizes = hidden_layers_sizes,
                                    loss= loss_i,
                                    optimizer='adam',  
                                    metrics=['mae'],

                                    n_folds = N_FOLDS,
                                    alpha = 1.e-2, #since CV search is done with alpha, this value does not matter but its necessary
                                    l_alpha = L_ALPHA,
                                    additional_hp_grid = additional_param_grid,
                                    cv_scoring_metric = 'neg_mean_absolute_error',

                                    #this is not early stopping(stop training if VALIDATION ERROR not decresing after N numerber of epochs)
                                    # this parameter means stop training if TRAINING ERROR is not decreasing at least min_delta after each epoch!
                                    use_tolerance = use_tolerance, 
                                    tolerance = tolerance,  
                                    patience = patience,  
                                    epochs = epochs,  
                                    batch_size = batch_size,

                                    seed = TRAIN_SEED,
                                    n_jobs = n_jobs,
                                    compress_level = 3,
                                    verbose = 0,
                                    debug = debug
                                )


                        ####################################
                        #   REPORT

                        saving_info = load_model(    os.path.join(save_path_i, MODEL_SAVING_NAME ) ,  
                                                                verbose = 0
                                                            )
                        update_log_status( debug= debug, report_folder_name = save_path, message = '\t\t\t-----> Generating report for LOSS:' + str(loss_i) )
                        np.set_printoptions(threshold=np.inf) # to print complete arrays

                        plot_cv_search_complete_report(      x,
                                                                        y,
                                                                        saving_info= saving_info,
                                                                        target_name = target_name,
                                                                        bins = N_BINS,
                                                                        n_cross_val_predict_execs = 5,
                                                                        n_cross_val_predict_seeds = VALIDATION_SEEDS,
                                                                        n_jobs = n_jobs,
                                                                        save_report= True,
                                                                        dataset_name = dataset_name_i, 
                                                                        save_path= save_path_i,
                                                                        save_name= REPORT_SAVING_NAME,
                                                                        debug = debug
                                                                )




        script_t_2 = time.time() 
        script_execution_time =  np.round( (script_t_2 - script_t_1)/60.  ,4) 
        update_log_status( debug= debug, report_folder_name = save_path, message = 'All work Done!' )
        update_log_status( debug= debug, report_folder_name = save_path, message = 'Complete Script Execution Time: ' + str(script_execution_time) + ' min.' )

                      




if __name__ == "__main__":
    main(sys.argv)