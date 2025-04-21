import os
import requests
import pandas as pd

from auth_app.models import UserProfile

API_KEY = UserProfile.api_key
BASE_URL = 'https://www.alphavantage.co/query'
output_size = 'compact'


def get_daily_time_series(function, symbol):
    params = {
        'function': function,
        'symbol': symbol,
        'outputsize': output_size,
        'apikey': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    if "Information" in data:
        return None, data["Information"]
    else:
        return data, None


def saving_files(json_data, filename):
    dataframe = pd.DataFrame.from_dict(json_data)
    directory = 'C:/Users/Ankush/PycharmProjects/Django/auth_app/files'
    file_path = os.path.join(directory, filename)
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    # Save the file using pandas
    dataframe.to_csv(file_path)


def getting_files(filename):
    file_path = 'C:/Users/Ankush/PycharmProjects/Django/auth_app/files/' + filename
    df = pd.read_csv(file_path, index_col=0)
    df.reset_index(inplace=True)
    return df

