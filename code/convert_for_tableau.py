import numpy as np
from sklearn.decomposition import PCA

from utilities import read_measurement_table

#Read the data (in 'tall' format)
meas_table_tall, battery_id_col_name, measure_id_col_name, freq_id_col_name,\
    impedance_col_name, soc_col_name = read_measurement_table('./config/config.yaml')

#Rearrange the data in 'wide' format-
primary_key = [measure_id_col_name, soc_col_name, battery_id_col_name]
meas_table_wide = meas_table_tall.pivot(primary_key, freq_id_col_name)
meas_table_wide = meas_table_wide.reset_index()
num_points = meas_table_wide.shape[0]

pca = PCA(n_components = 2)
modes = {'module' : [np.abs], 'phase' : [np.angle],
         'module+phase' : [np.abs, np.angle]}

for idx, (mode, functions) in enumerate(modes.items()):
    
    #Get the values (module, phase)
    values = np.zeros((num_points,0))
    for j, function in enumerate(functions):
        values = np.hstack((values,
                           function(meas_table_wide[impedance_col_name].to_numpy())))
    
    #Compute the first two principal components
    princomps = pca.fit_transform(values)