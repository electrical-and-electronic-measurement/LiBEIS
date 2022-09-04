"""Classification experiments. Estimate the accuracy of SOC prediction based
on impedance values"""
from itertools import product

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC

from tabulate import tabulate

from generics import classification_results_out, config_file
from utilities import Classifier, DataNormaliser, get_patterns,\
     FeatureExtractionMode, read_measurement_table

#Note: Some pattern_extraction_mode values are disabled to avoid  exceeding 
# the computation time limit of Code Ocean 
pattern_extraction_modes =\
    [FeatureExtractionMode(mode = 'module'),
     FeatureExtractionMode(mode = 'phase'),
     #FeatureExtractionMode(mode = 'module+phase'),
     #FeatureExtractionMode(mode = 'real'),
     #FeatureExtractionMode(mode = 'imag'),
     FeatureExtractionMode(mode = 'real+imag')]

classifiers =\
    [Classifier(name = 'Gaussian NB', model = GaussianNB, hyperparameters = None),
     Classifier(name = 'kNN', model = KNeighborsClassifier, hyperparameters = {'n_neighbors' : 1}),
     Classifier(name = 'kNN', model = KNeighborsClassifier, hyperparameters = {'n_neighbors' : 2}),
     Classifier(name = 'kNN', model = KNeighborsClassifier, hyperparameters = {'n_neighbors' : 3}),
     Classifier(name = 'lSVC', model = LinearSVC, hyperparameters = {'C' : 0.01, 'max_iter' : 10e4}),
     Classifier(name = 'lSVC', model = LinearSVC, hyperparameters = {'C' : 0.1, 'max_iter' : 10e4}),
     Classifier(name = 'lSVC', model = LinearSVC, hyperparameters = {'C' : 1.0, 'max_iter' : 10e4}),
     Classifier(name = 'lSVC', model = LinearSVC, hyperparameters = {'C' : 10.0, 'max_iter' : 10e4})]

normalization_modes =\
    [DataNormaliser(name = 'None', model = None),
     DataNormaliser(name = 'MinMax', model = MinMaxScaler),
     DataNormaliser(name = 'Z-score', model = StandardScaler)]

#Generate a full factorial plan by pattern_extraction_modes x classifiers
factorial_plan = product(pattern_extraction_modes, classifiers,
                         normalization_modes)
factorial_plan = list(factorial_plan)

#Remove non-normalised conditions for SVC classifier
to_remove = [x for x in factorial_plan if ((x[1].name == 'lSVC') & (x[2].name == 'None'))]
factorial_plan = [x for x in factorial_plan if x not in to_remove]

num_experiments = len(factorial_plan)

#Read the data
_, meas_table_wide, battery_id_col_name, freq_id_col_name, impedance_col_name,\
    measure_id_col_name, soc_col_name = read_measurement_table(config_file)

indices = np.arange(meas_table_wide.shape[0]).astype(np.uint)

#Store the results here
df_results = pd.DataFrame()

for experiment_idx, experiment in enumerate(factorial_plan):
    
    print(f'Running experiment {experiment_idx + 1} of {num_experiments}')
    
    predicted_labels = list()
    true_labels = list()
    
    #Compute the patterns
    patterns = get_patterns(meas_table_wide, impedance_col_name, 
                            mode = experiment[0].mode, 
                            kwargs = experiment[0].params)
    
    #Perform data normalisation
    patterns = experiment[2].normalise(patterns)
    
    #Estimate classification accuracy via 'protected' leave-one-out (the same 
    #battery cannot be both in the train and test set)
    for row_index in indices:
        
        #The leave-one-out condition
        leave_one_out_condition = (indices != row_index)
        
        #The no-same-battery condition
        battery_id = meas_table_wide.iloc[row_index][(battery_id_col_name)][0]
        no_same_battery_condition = np.array(\
            meas_table_wide[(battery_id_col_name)] != battery_id).astype(np.bool_)
        
        #Define the train and test set
        train_set_condition = leave_one_out_condition & no_same_battery_condition
        train_patterns = patterns[train_set_condition]
        train_labels = meas_table_wide[train_set_condition][(soc_col_name)].to_list()
        
        test_pattern = patterns[row_index].reshape(-1,1).T
        test_label = meas_table_wide.iloc[row_index][soc_col_name].to_list()[0]
        
        #Train the classifier and perform the classification
        experiment[1].train(train_patterns, train_labels)
        predicted_label = experiment[1].predict(test_pattern)[0]
        
        #Update the true and predicted labels
        predicted_labels.append(predicted_label)
        true_labels.append(test_label)

    #Estimate the accuracy
    acc = 100*accuracy_score(true_labels, predicted_labels)
    
    #Add record to dataframe
    record = pd.DataFrame({
        'Feature_extraction_mode' : experiment[0].mode,
        'Feature_normalisation_mode' : experiment[2].name,
        'Classifier' : experiment[1].name,
        'Classifier_hyperparameters' : str(experiment[1].hyperparameters),
        'Num_features' : patterns.shape[1],
        'Accuracy' : acc},
        index = [0])
    df_results = pd.concat([record, df_results.loc[:]]).reset_index(drop=True)

#Sort by accuracy
df_results = df_results.sort_values(['Accuracy'], ascending = [False])
    
with open(file = classification_results_out, mode = 'w') as fp:
    fp.write(tabulate(df_results, tablefmt = "simple", headers = "keys", 
                      floatfmt="3.1f"))

print("*=*=*= PROCESS COMPLETED =*=*=*")
print("SOC CLASSIFICATION MODEL SCORING RESULT EXPORTED IN RESULTS FOLDER. " +classification_results_out +"\n")

print("**____ THE END ____**")