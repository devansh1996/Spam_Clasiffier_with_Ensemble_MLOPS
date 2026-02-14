import pandas as pd
import os
import logging
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("english"))
ps = PorterStemmer()
encoder = LabelEncoder()
##Making Base Path

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root

train_path = BASE_DIR / "data" / "raw" / "train.csv"
test_path  = BASE_DIR / "data" / "raw" / "test.csv"



##Ensure the logs Directory exists
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

##Configuring a logger
logger=logging.getLogger('data_preprocess') #data_ingestion is the name of the logger
logger.setLevel('DEBUG') ## level of logger is debug which is first level

## setting console handler at level DEBUG
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

## setting file handler 
log_file_path=os.path.join(log_dir,'data_preprocess.log')##because we need to add info this file into the logging file
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

##Formatting 
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def transform_text(text):
    """TEXT TRANSFORMATION"""
    try:
        ps=PorterStemmer()
        text = text.lower()
        text = nltk.word_tokenize(text)
        text=[word for word in text if word.isalnum()]
        text=[word for word in text if word not in STOPWORDS and word not in string.punctuation ] 
        text=[ps.stem(word) for word in text]
        return " ".join(text)
    except Exception as e:
        logger.error(f'unable to feature engg{e}')
        raise
def preprocess(df:pd.DataFrame,text_column='text',target_column='target'):
    try:
        logger.debug('starting the process')
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')
        df=df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')
        df.loc[:,text_column]=df[text_column].apply(transform_text)
        logger.debug('text column transformed')
        return df
    except KeyError as e:
        logger.error('column not found:%s',e)
        raise
    except Exception as e:
        logger.error(f'Error during transformation{e}')

def main(text_column='text',target_column='target'):
    try:
        train_data=pd.read_csv(train_path)
        test_data=pd.read_csv(test_path)

        logger.debug('data loaded')
        train_data_preprocess=preprocess(train_data,text_column,target_column)
        test_preprocess=preprocess(test_data,text_column,target_column)
        ##store data inside processed folder
        data_path=os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)

        train_data_preprocess.to_csv(os.path.join(data_path,"train_preprocess.csv"),index=False)
        test_preprocess.to_csv(os.path.join(data_path,"test_preprocess.csv"),index=False)
        logger.debug('files saved')
    except Exception:
        logger.error('couldnt save')
    
if __name__=="__main__":
    main()
