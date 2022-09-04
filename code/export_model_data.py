
# import
import pandas as pd

def load_soc_dataset(measure_list,soc_list, dataset_path,CSV_FILE_PREFIX="FIT_MES",show_data=False):
  ''' Loads CVS file with equivalent circuit parameters fit data produced by acquisition system into a Pandas dataframe 
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
      df_rows=df_original.transpose()

      param_col_names= []
      for colIndex in range(0,df_rows.shape[1],1):
        param_col_names.append(param_list[colIndex])
      
      if show_data:
        print(param_col_names)
      
      df_rows.columns=param_col_names

      #for rowIndex, row in enumerate(df_rows):
      df_rows['SOC']=soc_list
      df_rows['EIS_ID']=measure_id
      df_rows['BATTERY_ID']=battery_id
      dataset= dataset.append(df_rows)
      if show_data:
        print(df_rows)

  return dataset,param_col_names

 
# Folder with source CSV files
dataset_path = "../results/"

# List of SoC level into dataset
soc_list=['100','090','080','070','060','050','040','030','020','010']
measure_list=["03_1","03_2","03_3","03_4","03_5"]
param_list=["R0", "R1", "Q1", "p1", "Q2", "p2", "L"]
#create Impedance Table file
dataset,feature_col_names=load_soc_dataset(measure_list,soc_list,dataset_path)
params_df= pd.DataFrame(columns=['MEASURE_ID','SOC','BATTERY_ID','PARAM_NAME','PARAM_VALUE'])

for param_name in param_list:
    #MEASURE_ID (type: string), SOC (type: float), BATTERY_ID (type: category), PARAM_NAME(type: string), PARAM_VALUE (type: Float)
    z_df= pd.DataFrame(columns = ['MEASURE_ID', 'BATTERY_ID', 'PARAM_NAME','PARAM_VALUE'])
    z_df['MEASURE_ID']=dataset['EIS_ID']
    z_df['SOC']=dataset['SOC']
    z_df['BATTERY_ID']=dataset['BATTERY_ID']
    z_df['PARAM_NAME']=param_name
    z_df['PARAM_VALUE']=dataset[param_name]

    params_df=pd.concat([params_df,z_df],ignore_index=True)

params_df.to_csv(dataset_path+"/params.csv",index=False)

print("*=*=*= PROCESS COMPLETED =*=*=*")
print("EQUIVALENT CIRCUIT PARAMETERS EXPORTED IN result FOLDER. params.csv \n")

