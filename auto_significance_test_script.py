#!/usr/bin/env python

''' Script wil perfom wilcoxon significance test
    over specified reports
'''

import os
import numpy as np
from sklearn.datasets import load_boston
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import  mean_absolute_error
import warnings
from scipy.stats import wilcoxon
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
from tempfile import mkdtemp
import os.path as path





CURRENT_DIRECTORY = r'G:\OneDrive\OneDrive - UAM\Sync\UAM\MASTER\TFM\CODIGO\REPORTS\_FINALES'
DATASETS_PATH = r'G:\OneDrive\OneDrive - UAM\Sync\UAM\MASTER\TFM\CODIGO\DATASETS'
SIGNIFICANCE_FOLDER_NAME = 'Model_CV_Predictions_and_Errors/'


N_BINS = 31 #Error distribution histogram
CLIP_PREDICTIONS = True
DEBUG = False       #if True re-print original raw taken predictions in each loss txt.


# dict {report_name: dataset_name}

REPORTS_NAMES_TO_USE = {    'boston_final_complete_19dic'  :  'boston_housing',
                            #'abalone_batch200'   :  'abalone' ,
                            #'cpu_batch200'   :  'cpusmall',
                            #'spacega_size50'   :  'space_ga' ,
                            #'mg_200'   :  'mg',
                            #'auto_mpg_batch_200'   :  'auto_mpg',
                            #'sotavento_size100_all'  :  'sotavento',
                }



          
LOSSES_TO_USE = [       'mse',
                        'mean_absolute_error',
                        'log_cosh',
                        'huber',
                        'pseudohuber', 
                        'e_insensitive',
                ]
'''
                        'cosine_similarity',
                        'quantile',
]
'''

LOSSES_SHORT_NAME = {   'mse' : 'MSE',
                        'mean_absolute_error' : 'MAE',
                        'log_cosh' : 'LogCosh',
                        'huber' : 'Huber',
                        'pseudohuber' : 'Pseudo-Huber',
                        'e_insensitive': 'e-insensitive'
                    }



DATASETS_TARGET_NAMES = {   'boston_housing' : 'MEDV',
                            'abalone' : 'Rings',
                            'cpusmall' : 'usr',
                            'space_ga' : 'Proporción_Votos_Emitidos',
                            'mg' : 'Mackey-Glass',
                            'auto_mpg': 'mpg',
                            'sotavento' : 'targ',
                        }





# Must specify min and max value
DATASETS_CLIP_RANGE = {     'boston_housing' : (5,50),
                            'abalone' : (0,30),
                            'cpusmall' : (0,100),
                            'space_ga' : (-np.inf, np.inf) ,
                            'mg' : (-np.inf, np.inf),
                            'auto_mpg': (-np.inf, np.inf),
                            'sotavento' : (0., 1.),
                        }



START_STRING =  'All predictions generated of all cross_val_predict_executions (l_cv_y_pred):'
END_STRING =  'GENERATING PLOTS WITH  MEAN PREDICTIONS!'

START_STRING_SOTAVENTO_DATASET =  'All predictions generated of all .predict executions (l_cv_y_pred) (TEST DATA):'
END_STRING_SOTAVENTO_DATASET = '-------------------------------------------------------------'
DELIMITADOR = 'dtype=float32),'





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

    #if(debug):
    #    print("loading from scikit datasets  ...")

    with warnings.catch_warnings():
        # You should probably not use this dataset.
        warnings.filterwarnings("ignore")
        boston = load_boston()

    x = boston['data']
    y = boston['target']

    return x,y, DATASETS_TARGET_NAMES['boston_housing']




#loads datset from libsvm file
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




#----------------------------SOTAVENTO ------------------------------------------

# Método para la reducción de la dimensionalidad de las coordenadas de Sotavento
def select_desired_variables(dataset):
    coordenadas = ["(43.625, -8.125)","(43.625, -8.0)","(43.625, -7.875)","(43.625, -7.75)","(43.625, -7.625)",
               "(43.5, -8.125)","(43.5, -8.0)","(43.5, -7.875)","(43.5, -7.75)","(43.5, -7.625)",
               "(43.375, -8.125)","(43.375, -8.0)","(43.375, -7.875)","(43.375, -7.75)","(43.375, -7.625)",
               "(43.25, -8.125)","(43.25, -8.0)","(43.25, -7.875)","(43.25, -7.75)","(43.25, -7.625)",
               "(43.125, -8.125)","(43.125, -8.0)","(43.125, -7.875)","(43.125, -7.75)","(43.125, -7.625)"]
    variables = ["10u_", "10v_", "2t_", "sp_", "100u_", "100v_", "vel10_", "vel100_"]
    
    indices = []
    for var in variables:
        for coord in coordenadas:
            indices.append(str(var) + str(coord))
    return(dataset[indices])


# Definición de una función para cargar los datos del problema Sotavento
def sotavento_data_test(dataset_path = DATASETS_PATH):
   
    sota2018_path = "data_target_stv_2018.csv"

    if(dataset_path):
        sota2018_path = os.path.join(dataset_path, sota2018_path)

    test = pd.read_csv(sota2018_path, sep = ',', index_col=0, parse_dates=True)
    x = select_desired_variables(test.drop(columns = ['targ'])).to_numpy()
    y = test[['targ']].to_numpy()

    return x, y.flatten()











def get_array_of_predictions(start_string, end_string ,file_path,out_path, min_clip, max_clip):
    content = ''
    with open(file_path, 'r') as file:  
        copy = False
        for line in file:
            if line.strip() == start_string:
                copy = True
                continue
            elif line.strip() == end_string:
                copy = False
                continue
            elif copy:
                content = content + line

    with open(out_path, 'w+') as file:  
        print( 'THIS FILE CONTAINS  cv_cross_val_predict() predictions FOR THIS CURRENT CONFIGURATION.\n\t\t It can be used to verificate obtained results.\n',file=file)

    if(DEBUG):
        with open(out_path, 'a+') as file:  
            print('\n\n\n-------------------------------------------------------------' , file = file)
            print( 'ORIGINAL ARRAY OF PREDICTIONS: \n',file=file)
            print(content,file=file)
            print('-------------------------------------------------------------\n\n\n\n\n\n' , file = file)


    content = content.replace('\n', '')
    content = content.replace(' ', '')
    content = content.replace('array([','')
    content = content.replace(']', '')


    d = DELIMITADOR
    list_of_arrays = [arr  for arr in content.split(d)]

    list_of_np_arrays_clipped = []


    # ISOLATING CV_PREDICTIONS
    if(DEBUG):
        with open(out_path, 'a+') as file:  
            print('\n\n-------------------------------------------------------------------------------------------------------------' , file = file)
            print( 'ISOLATED NUMPY ARRAYS OF PREDICTIONS (CLIPPED):',file=file)
            print('-------------------------------------------------------------------------------------------------------------' , file = file)

        

    i = 0
    for array in list_of_arrays:

        if(i == 0):
            array = array.replace('[', '')
        if(i == 4):
            array = array.replace('dtype=float32)', '')

            

        # CONVERTING TO NUMPY ARRAY
        l = array.split(',') 
        list_of_floats = [float(e) for e in l[:-1] ] 
        np_array_clipped = np.clip( np.array(list_of_floats) , min_clip,max_clip) 
        list_of_np_arrays_clipped.append(np_array_clipped)

        if(DEBUG):
            with open(out_path, 'a+') as file:  
                print('\n\n\n*************************************************************************************' , file = file)
                print( 'NUMPY ARRAY NUMBER:' + str(i+1) + '\n',file=file)
                print( np_array_clipped ,file=file)
                print('\n*************************************************************************************' , file = file)

        i = i+1

            


    with open(out_path, 'a+') as file:  

        a_cv_y_pred_clipped = np.array(list_of_np_arrays_clipped).T
        y_pred_by_cv_clipped = a_cv_y_pred_clipped.mean(axis=1)
        print('\n\n\n\n---------------------------------------------------------------------------------------------------------' , file = file)
        print( 'FINAL y_pred (clipped).  FINAL PREDICTIONS USED TO CALCULATE SCORE METRICS\n\t\t\t (mean over several cross_val_predict() executions)',file=file)
        print('---------------------------------------------------------------------------------------------------------' , file = file)
        print( y_pred_by_cv_clipped,file=file)


    return y_pred_by_cv_clipped





def plot_errs_histograms(save_path, u_name, v_name, u_errs, v_errs):


    plt.title("Difference between  " + u_name + ' and ' + v_name + ' errors (per prediction).' )
    plt.xlabel( '(' + u_name + ' - ' + v_name + ') errors.' )
    y_label = "Difference Value. ( " + str(N_BINS) + " bins used.)"
    plt.ylabel(y_label)
    _ = plt.hist(u_errs - v_errs, bins=N_BINS, color = "darkcyan")

    fig_name = 'ERRsDIFF__' + u_name + '__vs__' + v_name 
    plt.savefig(os.path.join(save_path,fig_name))
    plt.close()







def main():


    np.set_printoptions(threshold=np.inf) # to print complete arrays
    start_string, end_string = '',''

    
    for complete_report_name, dataset_name_i in REPORTS_NAMES_TO_USE.items():

        complete_report_path = os.path.join(CURRENT_DIRECTORY, complete_report_name )

        print('\n-----------------------------------------------------------------------------------------------------' )
        print('--> SWITCHING REPORT TO: ' + complete_report_name )
        print('-----------------------------------------------------------------------------------------------------\n\n' )
        print('-----> DATASET NAME: ' + dataset_name_i )

        #errors_by_loss = {}
        errors_by_loss_paths = {} #in disk paths of np arrays with each model cv predict errors
        cv_MAEs_by_loss = {}




        if(dataset_name_i == 'sotavento'):
            x,y =  sotavento_data_test()
            start_string = START_STRING_SOTAVENTO_DATASET
            end_string = END_STRING_SOTAVENTO_DATASET
        else:
            x,y, _ =  get_dataset(    dataset_name = dataset_name_i, 
                                                dataset_path = DATASETS_PATH, 
                                                debug = True
                                            )
            start_string = START_STRING
            end_string = END_STRING



        if(CLIP_PREDICTIONS):
            min_clip = DATASETS_CLIP_RANGE[dataset_name_i][0]
            max_clip= DATASETS_CLIP_RANGE[dataset_name_i][1]
        else:
            min_clip = - np.inf
            max_clip = np.inf


        #########################################################
        #  RECOVERING CV MAE SCORES
        ##########################################################

        for loss_i in LOSSES_TO_USE:

                y_pred_by_cv_clipped = np.empty(0)

                print('\t\t-----> USING NOW LOSS:' + loss_i )

                # Reading current 
                loss_folder_name = 'report__loss_' + str(loss_i) + '__dataset_' + str(dataset_name_i) 
                loss_report_path =  os.path.join(complete_report_path, loss_folder_name )
                loss_report_path_in =  os.path.join(loss_report_path, 'report.txt' )

                # Printing results in current folder
                loss_report_path_out =  os.path.join(loss_report_path, SIGNIFICANCE_FOLDER_NAME )
                os.makedirs(loss_report_path_out, exist_ok=True)
                loss_report_path_out_file =  os.path.join(loss_report_path_out, 'cv_predict_results.txt' )

          


                # Reading and getting model errs
                y_pred_by_cv_clipped = get_array_of_predictions(start_string, end_string ,loss_report_path_in, loss_report_path_out_file, min_clip, max_clip)

                # Temp array
                errs_temp_array_disk_path = os.path.join(loss_report_path_out, 'model_errs_cvpredict.data' )
                model_errors_ondisk = np.memmap(errs_temp_array_disk_path, dtype='float32', mode='w+', shape=y.shape)

                # Recalculating cv MAE Score
                model_mean_clipped = mean_absolute_error(y.ravel() , y_pred_by_cv_clipped)
                cv_MAEs_by_loss[loss_i] = model_mean_clipped

     
                model_errors_ondisk[:] = np.abs(y - y_pred_by_cv_clipped)
                errors_by_loss_paths[loss_i] = errs_temp_array_disk_path


                with open(loss_report_path_out_file, 'a+') as file:  
                    

                    print('\n\n\n*************************************************************' , file = file)
                    print('MODEL ERRORS (of CLIPPED predictions): \n ', file = file)
                    print(model_errors_ondisk, file = file)
                    print('\n*************************************************************\n' , file = file)

                    # Flushing memory
                    del model_errors_ondisk

                    
                    if(DEBUG):
                        print('\n\n\n-------------------------------------------------------------' , file = file)
                        print( '---> MODEL FINAL SCORE: CV MAE (clipped):',file=file)
                        print('-------------------------------------------------------------' , file = file)
                        print('MAE:  ' +  str(model_mean_clipped), file = file)


                    #  DEBUG
                    print('\t\t\t\t\t' + loss_i + '  SCORE:  ' + str(model_mean_clipped))


                print('\t\t Done!')



        #########################################################
        #                   Wilcoxon  TEST
        ##########################################################

        print("\t\t Executing Wilcoxon Tests...")

        # Test Print Path
        wilconxon_report_path =  os.path.join(complete_report_path, 'Wilcoxon_significance_tests/' )
        os.makedirs(wilconxon_report_path, exist_ok=True)
        wilconxon_txt_report_path = os.path.join(wilconxon_report_path, 'Wilcoxon_significance_tests_report.txt' )
            

        with open(wilconxon_txt_report_path, 'w+') as file:  

            # Reprinting cross_val_predict MAE Score per Model again...
            print('-------------------------------------------------------------' , file = file)
            print( 'cross_val_predict MAE Scores (CLIPPED) per Loss:',file=file)
            print('-------------------------------------------------------------' , file = file)
            for u,v in  cv_MAEs_by_loss.items():
                print(str(LOSSES_SHORT_NAME[u]) + ': \t' + str(v),file = file)
            print('\n\n\n' , file = file)



            # Wilcoxon
            for u in  list(errors_by_loss_paths.keys()):
                u_loss_errs = np.memmap(errors_by_loss_paths[u] , dtype='float32', mode='r', shape=y.shape)
                
                for v in  list(errors_by_loss_paths.keys()):
                
                
                    if(u == v):
                        continue

                    v_loss_errs = np.memmap(errors_by_loss_paths[v] , dtype='float32', mode='r', shape=y.shape)

                    print('\n\n*************************************************************' , file = file)
                    print('WILCOXON TEST:  ' + str(LOSSES_SHORT_NAME[u]) + ' vs. ' + str(LOSSES_SHORT_NAME[v]) + ' model errors', file = file)
                    print('*************************************************************' , file = file)

                    print( str(wilcoxon(u_loss_errs , v_loss_errs )), file = file)
                    print('\n\t---> PERCENTAGE OF TIMES ' +  str(LOSSES_SHORT_NAME[u]) + 'ERRRORS ARE GREATER THAN  ' + str(LOSSES_SHORT_NAME[v]) + ':' , file = file)
                    percentage = (u_loss_errs > v_loss_errs ).sum() / x.shape[0]
                    print('\t\t\t\t' + str( percentage), file = file)
                    print('\n\n\n' , file = file)


                    plot_errs_histograms(wilconxon_report_path, LOSSES_SHORT_NAME[u] , LOSSES_SHORT_NAME[v], u_loss_errs , v_loss_errs )

                    #Free mem
                    del v_loss_errs

                del u_loss_errs
                


        print("\t\t\t Done!\n")
        print("\t\t Cleaning temporal data...")
        for u in  errors_by_loss_paths.keys():
            os.remove(errors_by_loss_paths[u])

        print("\t\t\t Done!")



    print("Execution finished!")







if __name__ == "__main__":
    main()
