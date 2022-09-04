"""Generate two-dimensional scatter plots through PCA"""
from itertools import product
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns

from generics import config_file, pca_scatter_plots_out
from utilities import DataNormaliser, DataProjector, FeatureExtractionMode,\
     get_patterns, read_measurement_table

#Read the data
_, meas_table_wide, battery_id_col_name, freq_id_col_name, impedance_col_name,\
    measure_id_col_name, soc_col_name = read_measurement_table(config_file)

num_points = meas_table_wide.shape[0]

num_components = 2  #Number of components for the plots

soc_labels = meas_table_wide[soc_col_name]

#The feature extraction modes
pattern_extraction_modes =\
    [FeatureExtractionMode(mode = 'module'),
     FeatureExtractionMode(mode = 'phase'),
     FeatureExtractionMode(mode = 'module+phase'),
     FeatureExtractionMode(mode = 'real'),
     FeatureExtractionMode(mode = 'imag'),
     FeatureExtractionMode(mode = 'real+imag')]

#The data normalisation modes
#normalization_modes =\
    #[DataNormaliser(name = 'None', model = None),
     #DataNormaliser(name = 'MinMax', model = MinMaxScaler),
     #DataNormaliser(name = 'Z-score', model = StandardScaler)]

normalization_modes =\
    [DataNormaliser(name = 'None', model = None)]

#The feature transformation models (LDA, PCA, etc.)
projection_modes = [DataProjector('LDA (svd)', LDA),
                    DataProjector('PCA', PCA, {'n_components' : num_components})]


#Combinations feature extraction/feature transformation
combinations = product(pattern_extraction_modes, normalization_modes,
                       projection_modes)
combinations = list(combinations)

#Figure layout
ncols = len(pattern_extraction_modes)
nrows = np.ceil(len(combinations)/ncols).astype(np.int8)
fig = plt.figure()
fig.set_size_inches(60, 20, forward=True)

for idx, combination in enumerate(combinations):
    
    #feature_name, feature_functions = combination[0]
    #transformation_name, transformation_function = combination[1]
    
    ax = fig.add_subplot(nrows, ncols, idx + 1)
    
    #Get the raw feature values
    mode = combination[0].mode
    values = get_patterns(meas_table_wide, impedance_col_name, mode)
    
    #Apply the normalisation
    values = combination[1].normalise(values)
        
    #Compute the first two principal components
    princomps = combination[2].project(data_in = values, 
                                       class_labels = soc_labels)
    
    #Place the legend on the last plot only
    legend = False
    if idx == (len(combinations) - 1):
        legend = 'full'
    
    #Draw the scatter plot
    sns.scatterplot(x = princomps[:,0], y = princomps[:,1], ax = ax,
                    hue = meas_table_wide[soc_col_name],
                    style = meas_table_wide[battery_id_col_name],
                    palette = 'RdYlBu_r', legend = legend,
                    edgecolor = 'black'
                    )
    ax.set_title(f'Feature: {combination[0].mode}, Normalisation: {combination[1].name}, Projection: {combination[2].name}', 
                 fontsize = 10)
    ax.set_xlabel('1st principal component', fontsize = 10)
    ax.set_ylabel('2nd principal component', fontsize = 10)
    ax.grid()
    if legend:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

fig.subplots_adjust(wspace = 0.4, hspace = 0.4)
fig.savefig(fname = pca_scatter_plots_out)

print("*=*=*= PROCESS COMPLETED =*=*=*")
print("LDA / PCA PLOTS EXPORTED IN result FOLDER. " +pca_scatter_plots_out +"\n")
