
# import
import pandas as pd

def load_soc_dataset(measure_list,soc_list, dataset_path,CSV_FILE_PREFIX="EIS_BATT",show_data=False):
  ''' Loads CVS file with EIS data produced by acquisition system into a Pandas dataframe 
  Args: CSV_FILE_PREFIX: the prefix of the CSV file to load , dataset_path: the path where the CSV files are stored, 
  measure_list: the list of measures to load (MEASURE_ID), soc_list: the list of SOCs in the source files, show_data: if True, the data is displayed '''
  dataset = pd.DataFrame(columns=['SOC','BATTERY_ID','EIS_ID'])
  for measure_index, measure_id in enumerate(measure_list):
    battery_id=measure_id.split("_")[0]
    if show_data:
      print("measure_id: "+str(measure_id))
      print("battery_id: "+str(battery_id))
    #Create a Pandas dataframe from CSV
    df_original= pd.read_csv(dataset_path+CSV_FILE_PREFIX+str(measure_id)+"_ALL_SOC.csv",names=soc_list, low_memory=False)
    #note: csv from matlab are in format 12-64i.
    #      'i" must be replaced with "j" into the CVS file
    df = df_original.apply(lambda col: col.apply(lambda val: val.replace('i','j')))
    #Parse complex number in format: 123-56j, 432+56j
    df = df.apply(lambda col: col.apply(lambda val: complex(val)))
    df_rows=df.transpose()

    eis_col_names= []
    for colIndex in range(0,df_rows.shape[1],1):
      eis_col_names.append("Z_f"+str(colIndex))
    
    if show_data:
      print(eis_col_names)
    
    df_rows.columns=eis_col_names

    #for rowIndex, row in enumerate(df_rows):
    df_rows['SOC']=soc_list
    df_rows['EIS_ID']=measure_id
    df_rows['BATTERY_ID']=battery_id
    dataset= dataset.append(df_rows)
    if show_data:
      print(df_rows)

  return dataset,eis_col_names

 
# Folder with source CSV files
dataset_path = "../results/"

# List of SoC level into dataset
soc_list=['100','090','080','070','060','050','040','030','020','010']
frequency_list=[0.05, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40, 100, 200, 400, 1000]
measure_list=["03_1","03_2","03_3","03_4","03_5"]

#create Frequency Table file
frequencies_df=pd.DataFrame({'FREQUENCY_ID':range(len(frequency_list)),'FREQUENCY_VALUE':frequency_list})
frequencies_df.to_csv(dataset_path+"frequencies.csv",index=False)

#create Impedance Table file
dataset,feature_col_names=load_soc_dataset(measure_list,soc_list,dataset_path)
impedance_df= pd.DataFrame(columns=['MEASURE_ID','SOC','BATTERY_ID','FREQUENCY_ID','IMPEDANCE_VALUE'])

for freq_id in frequencies_df['FREQUENCY_ID']:
    #MEASURE_ID (type: string), SOC (type: float), BATTERY_ID (type: category), FREQUENCY_ID(type: string), IMPEDANCE_VALUE (type: Complex Float)
    z_df= pd.DataFrame(columns = ['MEASURE_ID', 'BATTERY_ID', 'FREQUENCY_ID','IMPEDANCE_VALUE'])
    z_df['MEASURE_ID']=dataset['EIS_ID']
    z_df['SOC']=dataset['SOC']
    z_df['BATTERY_ID']=dataset['BATTERY_ID']
    z_df['FREQUENCY_ID']=str(freq_id)
    z_df['IMPEDANCE_VALUE']=dataset['Z_f'+str(freq_id)]

    impedance_df=pd.concat([impedance_df,z_df],ignore_index=True)

impedance_df.to_csv(dataset_path+"impedance.csv",index=False)
print("*=*=*= PROCESS COMPLETED =*=*=*")
print("DATA ANALYSIS PLOTA EXPORTED IN result FOLDER. OCV_dispersion.pdf , SOC_vs_OCV.pdf, EIS_curves.pdf \n")
print("EIS DATA EXPORTED IN impedence.csv and frequency.csv FILES IN result FOLDER \n")



