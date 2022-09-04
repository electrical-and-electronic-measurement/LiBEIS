"""General settings"""
import yaml

config_file = './config/config.yaml'

#Read the configuration
with open(config_file) as cfg_file:
    config = yaml.load(stream = cfg_file, Loader = yaml.FullLoader)

classification_results_out = config['classification_results_out']
pca_scatter_plots_out = config['pca_scatter_plots_out']