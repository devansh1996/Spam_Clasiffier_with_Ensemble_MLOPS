import pandas as pd
import os 
from sklearn.model_selection import train_test_split
import logging
import yaml
import chardet
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # src/.. = repo root
DATA_DIR = PROJECT_ROOT / "data"

##Ensure the logs Directory exists
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

##Configuring a logger
logger=logging.getLogger('data_ingestion') #data_ingestion is the name of the logger
logger.setLevel('DEBUG') ## level of logger is debug which is first level

## setting console handler at level DEBUG
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

## setting file handler 
log_file_path=os.path.join(log_dir,'data_ingestion.log')##because we need to add info this file into the logging file
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

##Formatting 
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def load_data(url):
    try:
        df = pd.read_csv(url, encoding='latin-1')
        return df
    except pd.errors.ParserError as e:
     logger.error('Failed to parse the csv file:%s',e)
     raise
    except Exception as e:
      logger.error("Unexpected error occured while loading data:%s",e)
      raise

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
   """preprocess data """
   try:
      df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
      df.rename(columns={'v1':'target','v2':'text'},inplace=True)
      logger.debug('Data preprocessing completed')
      return df
   except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
   except Exception as e:
      logger.error('Unexpected error during preprocessing: %s', e)
      raise
   

  
def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str):
    """Save the train and test datasets."""
    try:
        raw_dir = (Path(data_path) / "raw").resolve()
        raw_dir.mkdir(parents=True, exist_ok=True)

        train_file = raw_dir / "train.csv"
        test_file  = raw_dir / "test.csv"

        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        logger.debug("Train saved to: %s", train_file)
        logger.debug("Test saved to: %s", test_file)
        logger.debug("Raw dir exists after save: %s", raw_dir.exists())
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise
   
def main():
   try:
    test_size=0.25
    #   params = load_params(params_path='params.yaml')
    # test_size = params['data_ingestion']['test_size']
    data_path='https://raw.githubusercontent.com/devansh1996/Spam_Clasiffier_with_Ensemble_MLOPS/main/spam.csv'
    df=load_data(data_path)
    final_df=preprocess_data(df)
    train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=0000)
    save_data(train_data,test_data,data_path=str(DATA_DIR)) ##. mean here to go to root
   except Exception as e:
        logger.error('failed in data load:%s',e)
        raise

if __name__=='__main__':
   main()

      
      

