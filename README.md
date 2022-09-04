# LiBEIS

Software tool for Electrochemical Impedance Spectroscopy (EIS) computation on rechargable litium batteries

## Data

Before run LiBEIS you need some EIS data. You can downaload EIS data from: [Dataset on broadband Electrochemical Impedance Spectroscopy of Lithium-Ion Batteries for Different Values of the State of Charge](https://data.mendeley.com/drafts/mbv3bx847g).

```
Buchicchio, Emanuele; De Angelis, Alessio; Santoni, Francesco; Carbone, Paolo (2022), “Dataset on broadband Electrochemical Impedance Spectroscopy of Lithium-Ion Batteries for Different Values of the State of Charge”, Mendeley Data, V3, doi: 10.17632/mbv3bx847g.3
```

Download `imepdance.csv` and `frequencyes.csv` into the data folder.

## Usage

1. run `matlab -nodisplay -r "addpath(genpath('.'))`; to compute_impedance_values.
1. run `python3 /code/export_eis_data.py` to generate the EIS dataset files (impedence.csv and frequency.csv). A copy of these files can also be download from  https://data.mendeley.com/drafts/mbv3bx847g
1. run `python3 /code/lda_pca_scatter_plots.py`to Perform LDA and PCA and save the scattering plots in reults folder
1. run `python3 /code/classification.py` to train and score different SOC classification models. retrieve the results in the `classification_results_out@config/config.yaml` file

1. run `matlab -nodisplay -r "addpath(genpath('.'))` to fit the equivalent circuit model of the battery
1. run `python3 /code/export_model_data.py` to generate the circuit model parameters dataset file (parameters.csv)

## Settings

- Edit the `config/config.yaml` to adjust the settings - e.g., connection to the data source and output folders.
