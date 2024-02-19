import cmath
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

AUGMENTATION_MEAS_ID_OFFSET=1000

def complex_array_from_real_imag(real, imag):
    """Convert a pair of arrays of real and imaginary values into a complex array
    
    Parameters
    ----------
    real : array-like
        Array of real values
    imag : array-like
        Array of imaginary values
        
    Returns
    -------
    complex_array : array-like
        Array of complex values
    """
    complex_array = real + 1j * imag
    return complex_array

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
    with open(config_file) as cfg_file:
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

def augment_meas_data(meas_table_wide, impedance_col_name, meas_id_col_name ,data_augmentation_factor,noise_std_dev):
    """Augment the measurement data by adding noise to the impedance values
    
    Parameters
    ----------
    meas_table_wide : pd.DataFrame
        The measurement table in 'wide' format as returned by 
        read_measurement_table()
    impedance_col_name : str
        Name of the field that identifies the impedance value
    data_augmentation_factor : int
        Factor by which the data is augmented
    """
    #Add noise to the impedance values
    impedance_values = meas_table_wide[impedance_col_name].to_numpy()

    augmented_mes_table_wide=meas_table_wide.copy()

    for augmentation_index in range(1,data_augmentation_factor):
        noise_cplx = np.random.normal(0, noise_std_dev, impedance_values.shape) +np.random.normal(0, noise_std_dev, impedance_values.shape)*1j
        meas_table_wide_copy = meas_table_wide.copy()
        meas_table_wide_copy[impedance_col_name] = impedance_values + noise_cplx
        #get the original measure id vector
        original_meas_id=meas_table_wide_copy[meas_id_col_name]
        
        #change the measure id in format <battery_id>_<measure_num> replacing the measure id with the augmented measure id    
        augment_meas_id_col=[]
        for i in original_meas_id.index:
           original_battery_id=original_meas_id[i].split('_')[0]
           original_meas_num=original_meas_id[i].split('_')[1]
           augmented_measure_num=int(original_meas_num)*AUGMENTATION_MEAS_ID_OFFSET+augmentation_index

           augmented_measure_id=original_battery_id+'_'+str(augmented_measure_num)           
           augment_meas_id_col.append(augmented_measure_id)
        # replace the measure id column with the augmented measure id column
        meas_table_wide_copy[meas_id_col_name]=augment_meas_id_col

    
        #append the augmented data to the original data
        augmented_mes_table_wide = pd.concat([augmented_mes_table_wide, meas_table_wide_copy], ignore_index=True)

    
    return augmented_mes_table_wide


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
    elif mode == 'bode':
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
def generate_synthetic_xy_values(patterns_row, mode,noise_std_dev):
    """Generate synthetic data for 2D visual representation.
    parameters:
    patterns_row: array of feature values extracted with the method get_patterns()
    mode: str A string determining the way the visualization are computed from the
        impedance values. Can be:
            'module' -> x_values: index; y_values are the modules (abs) of the impedance values
            'phase'  -> x_values: index; y_values are the phase (angle) of the impedance values
            'module+phase' -> x_values: the modules (abs) of the impedance values; y_values the phase (angle) of the impedance values
            'imag' -> x_values: index; y_values are the imaginary part  of the impedance values
            'real' -> px_values: index; y_values are the real part of the impedance values
            'real+imag' -> x_values: the real part of the impedance values; y_values the imaginary part of the impedance values
            'bode' -> x_values: the modules (abs) of the impedance values; y_values the phase (angle) of the impedance values
    noise_std_dev: float standard deviation of the random noise to be added to the original data to generate the synthetic data
    """
    x_values = None
    y_values = None
    y2_values = None
    if mode == 'module':
        cplx_values = []
        for i in range(len(patterns_row)):
            cplx_values.append(cmath.rect(patterns_row[i],0))
        
        real_with_noise= np.real(cplx_values) + np.random.normal(0,noise_std_dev,len(cplx_values))    
        imag_with_noise= np.imag(cplx_values) + np.random.normal(0,noise_std_dev,len(cplx_values))
        cplx_values_with_noise = complex_array_from_real_imag(real_with_noise, imag_with_noise)
        x_values = np.arange(len(patterns_row))
        y_values = np.abs(cplx_values_with_noise)
    elif mode == 'phase':
        cplx_values = []
        for i in range(len(patterns_row)):
            cplx_values.append(cmath.rect(1,patterns_row[i]))        
        real_with_noise= np.real(cplx_values) + np.random.normal(0,noise_std_dev,len(cplx_values))    
        imag_with_noise= np.imag(cplx_values) + np.random.normal(0,noise_std_dev,len(cplx_values))
        cplx_values_with_noise = complex_array_from_real_imag(real_with_noise, imag_with_noise)
        x_values = np.arange(len(patterns_row))
        y_values = np.angle(cplx_values_with_noise)        
    elif mode == 'module+phase':
        abs_values = patterns_row[:len(patterns_row)//2]
        phase_values = patterns_row[len(patterns_row)//2:]
        cplx_values = []
        for i in range(len(abs_values)):
            cplx_values.append(cmath.rect(abs_values[i],phase_values[i]))
        real_with_noise= np.real(cplx_values) + np.random.normal(0,noise_std_dev,len(cplx_values))
        imag_with_noise= np.imag(cplx_values) + np.random.normal(0,noise_std_dev,len(cplx_values))
        cplx_values_with_noise = complex_array_from_real_imag(real_with_noise, imag_with_noise)
        x_values = np.abs(cplx_values_with_noise)
        y_values = np.angle(cplx_values_with_noise)

    elif mode == 'bode':
        x_values = np.arange(len(patterns_row)//2)
        abs_values = patterns_row[:len(patterns_row)//2]
        phase_values = patterns_row[len(patterns_row)//2:]
        cplx_values = []
        for i in range(len(abs_values)):
            cplx_values.append(cmath.rect(abs_values[i],phase_values[i]))
        real_with_noise= np.real(cplx_values) + np.random.normal(0,noise_std_dev,len(cplx_values))
        imag_with_noise= np.imag(cplx_values) + np.random.normal(0,noise_std_dev,len(cplx_values))
        cplx_values_with_noise = complex_array_from_real_imag(real_with_noise, imag_with_noise)
        y_values = np.abs(cplx_values_with_noise)
        y2_values = np.angle(cplx_values_with_noise)
    elif mode == 'imag':
        x_values = np.arange(len(patterns_row))
        y_values = patterns_row + np.random.normal(0,noise_std_dev,len(patterns_row))
        
    elif mode == 'real':
        x_values = np.arange(len(patterns_row))
        y_values = patterns_row + np.random.normal(0,noise_std_dev,len(patterns_row))
    elif mode == 'real+imag':
        x_values = patterns_row[:len(patterns_row)//2] + np.random.normal(0,noise_std_dev,len(patterns_row)//2)
        y_values = patterns_row[len(patterns_row)//2:] + np.random.normal(0,noise_std_dev,len(patterns_row)//2)
    else:
        raise Exception(f'Pattern calculation mode *{mode}* not supported')
    
    #add noise to the data
    x_values = x_values + np.random.normal(0,noise_std_dev,len(x_values))
    y_values = y_values + np.random.normal(0,noise_std_dev,len(y_values))
    
    return x_values, y_values,y2_values

def get_xy_values(pattern_row, mode):
    """Get visual 2D visual representation the features extracted from the impedance values of an EIS measurement.
    
    Parameters
    ----------
    pattern_row : array of feature values extracted with the method get_patterns()
    mode : str
        A string determining the way the visualization are computed from the
        impedance values. Can be:
            'module' -> x_values: index; y_values are the modules (abs) of the impedance values
            'phase'  -> x_values: index; y_values are the phase (angle) of the impedance values
            'module+phase' -> x_values: the modules (abs) of the impedance values; y_values the phase (angle) of the impedance values
            'imag' -> x_values: index; y_values are the imaginary part  of the impedance values
            'real' -> px_values: index; y_values are the real part  of the impedance values
            'real+imag' -> x_values: the real part of the impedance values; y_values the imaginary of the impedance values
    """
    n_cols = len(pattern_row)
    x_values = []
    y_values = []
    y2_values = []
    
    if mode == 'module':
        for col_index in range(0, n_cols):
            x_values.append(col_index)
            y_values.append(pattern_row[col_index])
    elif mode == 'phase':
        for col_index in range(0, n_cols):
            x_values.append(col_index)
            y_values.append(pattern_row[col_index])
    elif mode == 'module+phase':
        for col_index in range(0, int(n_cols/2)):
            x_values.append(pattern_row[col_index])
        for col_index in range(int(n_cols/2), n_cols):
            y_values.append(pattern_row[col_index])
    elif mode == 'imag':
        for col_index in range(0, n_cols):
            x_values.append(col_index)
            y_values.append(pattern_row[col_index])
    elif mode == 'real':
        for col_index in range(0, n_cols):
            x_values.append(col_index)
            y_values.append(pattern_row[col_index])
    elif mode == 'real+imag':
        for col_index in range(0, int(n_cols/2)):
            x_values.append(pattern_row[col_index])
        for col_index in range(int(n_cols/2), n_cols):
            y_values.append(pattern_row[col_index])
    elif mode == 'bode':
        for col_index in range(0, int(n_cols/2)):
            x_values.append(col_index)
            y_values.append(pattern_row[col_index])
        for col_index in range(int(n_cols/2), n_cols):
            y2_values.append(pattern_row[col_index])
    else:
        raise Exception(f'Pattern calculation mode *{mode}* not supported')
    return [x_values,y_values,y2_values]

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
        
    