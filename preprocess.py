import pandas as pd
import numpy as np
from Config import Config
import re

def get_input_data() -> pd.DataFrame:
    # Load both CSV files to get a rich dataset
    df1 = pd.read_csv('data/AppGallery.csv')
    df2 = pd.read_csv('data/Purchasing.csv')
    df = pd.concat([df1, df2], ignore_index=True)
    # Fill NA values to prevent issues in string operations
    df[Config.TYPE_COLS] = df[Config.TYPE_COLS].fillna('')
    return df

def de_duplication(df: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicates based on Interaction content
    return df.drop_duplicates(subset=[Config.INTERACTION_CONTENT]).reset_index(drop=True)

def noise_remover(df: pd.DataFrame) -> pd.DataFrame:
    # Basic noise removal: lowercase, remove non-alphanumeric, fillna
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].fillna('')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].fillna('')
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x).lower()))
    return df

def translate_to_en(text_list: list) -> list:
    # Returning the original list for now to save processing time.
    # If translation is strictly required, we can use the googletrans library:
    # from googletrans import Translator
    # translator = Translator()
    # return [translator.translate(str(text), dest='en').text for text in text_list]
    return text_list