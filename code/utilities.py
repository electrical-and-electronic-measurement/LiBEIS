from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

def read_measurement_table(config_file):
    """Read the measurement table
    
    Parameters
    ----------
    config_file : str
        Pointer to the yaml file where the configuration data are stored (e.g.,
        pointer to the source file and table structure)
        
    Returns
    -------
    meas_table_tall : pd.DataFrame
        The measurement table in 'tall' format
    meas_table_wide : pd.DataFrame
        The multi-index measurement table in 'wide' format
    battery_id_col_name :
        Name of the field that identifies the battery id
    freq_id_col_name : str
        Name of the field that identifies the frequency id
    impedance_col_name : str
        Name of the field that identifies the impedance value
    measure_id_col_name : str
        Name of the field that identifies the measure id
    soc_col_name : str
        Name of the field that identifies the state of charge
    """
    
    #Read the configuration
    with open('./config/config.yaml') as cfg_file:
        config = yaml.load(stream = cfg_file, Loader = yaml.FullLoader)
    
    #Map the field names
    battery_id_col_name = config['battery_id_field'] 
    freq_id_col_name = config['frequency_id_field']
    impedance_col_name = config['impedance_field']    
    measure_id_col_name = config['measure_id_field']  
    soc_col_name = config['soc_field']
    
    #Load the measurement table
    meas_table_tall = pd.read_csv(config['meas_table_src'])
    
    #Parse the impedance and convert it to complex
    impedance_col_name = config['impedance_field']
    impedance = meas_table_tall[impedance_col_name].to_numpy()
    impedance = list(map(complex, impedance))  
    meas_table_tall[impedance_col_name] = impedance
    
    #Rearrange the data in 'wide' format-
    primary_key = [measure_id_col_name, soc_col_name, battery_id_col_name]
    meas_table_wide = meas_table_tall.pivot(primary_key, freq_id_col_name)
    meas_table_wide = meas_table_wide.reset_index()   
    
    return meas_table_tall, meas_table_wide, battery_id_col_name,\
           freq_id_col_name, impedance_col_name, measure_id_col_name,\
           soc_col_name

def get_patterns(meas_table_wide, impedance_col_name, mode, **kwargs):
    """Get the pattern data from the impedance values
    
    Parameters
    ----------
    meas_table_wide : pd.DataFrame
        The measurement table in 'wide' format as returned by 
        read_measurement_table()
    impedance_col_name : str
        Name of the field that identifies the impedance value
    mode : str
        A string determining the way the patterns are computed from the
        impedance values. Can be:
            'module' -> patterns are the modules (abs) of the impedance values
            'phase'  -> patterns are the phase (angle) of the impedance values
            'module+phase' -> both of the above
            'imag' -> patterns are the imaginary part of the impedance values
    """
    patterns = None
    impedance_values = meas_table_wide[impedance_col_name].to_numpy()
    
    if mode == 'module':
        patterns = np.abs(impedance_values)   
    elif mode == 'phase':
        patterns = np.angle(impedance_values)
    elif mode == 'module+phase':
        patterns = np.hstack((np.abs(impedance_values),
                              np.angle(impedance_values)))
    elif mode == 'imag':
        patterns = np.imag(impedance_values)
    elif mode == 'real':
        patterns = np.real(impedance_values)
    elif mode == 'real+imag':
        patterns = np.hstack((np.real(impedance_values),
                              np.imag(impedance_values)))        
    else:
        raise Exception(f'Pattern calculation mode *{mode}* not supported')
    
    return patterns


@dataclass
class FeatureExtractionMode:
    """Class for encapsulating the way the patterns are computed from the
    impedance values"""
    
    #The value of the parameter 'mode' passed to get_patterns()
    mode: str 
    
    #A dict containing the optional parameters (if any) passed to get_patterns(
    #) through kwargs
    params: dict = None
    
class DataHandler():
    """Base class for DataNormaliser and DataProjector"""
    
    def __init__(self, name, model, params = None):
        """Initialise an empty normaliser/projector
        
        Parameters
        ----------
        name : str
            A user-friendly name for the normaliser
        model : object
            The normalisation/projection model. Pass None if no normalisation/ 
            projection is requested.
        params : optional arguments
        """
        
        self._name = name
        self._model = model
        if self._model:
            if params:
                self._model = self._model(**params)
            else:
                self._model = self._model()
        a = 0
    
    @property
    def name(self):
        return self._name    
     
class DataNormaliser(DataHandler):
    """Class for encapsulating data normalisation"""
            
    def normalise(self, data_in):
        """Perform data normalisation
        
        Parameters
        ----------
        data_in : array-like (N, M)
            An N x M matrix where each row represents one observation and each
            column one feature
        
        Returns
        -------
        data_out : array-like (N, M)
            The normalised data
        """
        data_out = data_in.copy()
        if self._model:
            data_out = self._model.fit_transform(X = data_in)
            
        return data_out
    
class DataProjector(DataHandler):
    """Class for encapsulating the way the data normalisation method"""
            
    def project(self, data_in, class_labels):
        """Perform data normalisation
        
        Parameters
        ----------
        data_in : array-like (N, M)
            An N x M matrix where each row represents one observation and each
            column one feature
        class_labels : array-like (N)
            The class labels
        
        Returns
        -------
        data_out : array-like (N, num_components)
            The projected data
        """
        data_out = data_in.copy()
        if self._model:
            data_out = self._model.fit_transform(X = data_in, y = class_labels)
            
        return data_out   
    
    
class Classifier:
    """Class for encapsulating a classifier"""
    
    def __init__(self, name, model, hyperparameters):
        """Initialise an empty classification model
        
        Parameters
        ----------
        name : str
            A user-friendly name for the classifier
        model : object
            The classification model
        hyperparameters : dict
            Th hyperparameters
        """
        
        self._name = name
        self._hyperparameters = hyperparameters
        
        if hyperparameters:
            self._model = model(**hyperparameters)
        else:
            self._model = model()
        
    @property
    def name(self):
        return self._name
    
    @property
    def hyperparameters(self):
        return self._hyperparameters
        
    def train(self, train_patterns, train_labels):
        """Train the classifier
        
        Parameters
        ----------
        train_patterns : array-like (N, M)
            An N x M matrix where each row represents one observation and each
            column one feature
        train_labels : array-like (N,)
            The train labels
        """
        self._model.fit(train_patterns, train_labels)
        
    def predict(self, patterns_to_classify):
        """Perform the predictions
        
        Parameters
        ----------
        patterns_to_classify : array-like (N, M)
            The patterns to classify
        
        Returns
        -------
        predicted_labels : array-like (N,)
            The predicted labels
        """
        return self._model.predict(patterns_to_classify)
        
    