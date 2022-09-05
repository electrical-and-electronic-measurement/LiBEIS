# LiBEIS

LiBEIS is a software tool for Electrochemical Impedance Spectroscopy (EIS) computation on rechargable litium batteries (LiB)

See also [https://electrical-and-electronic-measurement.github.io/LiBEIS/](https://electrical-and-electronic-measurement.github.io/LiBEIS/)

## Data

Before run LiBEIS you need some EIS data. 

To test the EIS computation Matlab script you can downalod some raw Voltage and current data acquisition file from [Dataset on Voltage and Current Data Acquisition During Broadband Electrochemical Impedance Spectroscopy of Lithium-Ion Batteries for Different Values of the State of Charge](https://data.mendeley.com/datasets/zdsgxwksn5)

```
Buchicchio, Emanuele; De Angelis, Alessio; Santoni, Francesco; Carbone, Paolo (2022), “Dataset on Voltage and Current Data Acquisition During Broadband Electrochemical Impedance Spectroscopy of Lithium-Ion Batteries for Different Values of the State of Charge”, Mendeley Data, V1, doi: 10.17632/zdsgxwksn5.1
```

Dowalod at least on battery folder (including all subfolders) into `/data` folder

You can also downaload EIS data from: [Dataset on broadband Electrochemical Impedance Spectroscopy of Lithium-Ion Batteries for Different Values of the State of Charge](https://data.mendeley.com/datasets/mbv3bx847g).

```
Buchicchio, Emanuele; De Angelis, Alessio; Santoni, Francesco; Carbone, Paolo (2022), “Dataset on broadband Electrochemical Impedance Spectroscopy of Lithium-Ion Batteries for Different Values of the State of Charge”, Mendeley Data, V3, doi: 10.17632/mbv3bx847g.3
```

Download `imepdance.csv` and `frequencyes.csv` into the `/data` folder.

## Usage

There are two options to use LiBEIS:

1. from a docker container image (see the docker file in `/environment` folder)
1. locally from the `/code` folder:

### Run Locally 

1. Move to `/code`
2. run `matlab -nodisplay -r "addpath(genpath('.'))`; to compute_impedance_values.
3. run `python3 export_eis_data.py` to generate the EIS dataset files (impedence.csv and frequency.csv). A copy of these files can also be download from  https://data.mendeley.com/datasets/mbv3bx847g
4. run `python3 lda_pca_scatter_plots.py`to Perform LDA and PCA and save the scattering plots in reults folder
5. run `python3 classification.py` to train and score different SOC classification models. retrieve the results in the `classification_results_out@config/config.yaml` file

6. run `matlab -nodisplay -r "addpath(genpath('.'))` to fit the equivalent circuit model of the battery
7. run `python3 export_model_data.py` to generate the circuit model parameters dataset file (parameters.csv)

## Settings

- Edit the `config/config.yaml` to adjust the settings - e.g., connection to the data source and output folders.
