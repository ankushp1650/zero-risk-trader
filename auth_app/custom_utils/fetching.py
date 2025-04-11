import os
import requests
import pandas as pd

from auth_app.models import UserProfile

API_KEY = UserProfile.api_key  # 'HA3VIAC6G4EQNEZS' 'HSSV9C1COJK3YMP5' #'HSSV9C1COJK3YMP5' #
BASE_URL = 'https://www.alphavantage.co/query'
output_size = 'compact' #'full'


def get_daily_time_series(function, symbol):
    params = {
        'function': function,
        'symbol': symbol,
        'outputsize': output_size,
        'apikey': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    # filename = str(function) + '_' + str(symbol) + '.csv'
    print(data)
    # return data
        #, function, symbol, filename
    if "Information" in data:
        return None, data["Information"]  # Return None for data and the message
    else:
        return data, None


def saving_files(json_data, filename):
    dataframe = pd.DataFrame.from_dict(json_data)
    # print(dataframe.to_string())
    directory = 'C:/Users/Ankush/PycharmProjects/Django/auth_app/files'
    file_path = os.path.join(directory, filename)
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    # Save the file using pandas
    dataframe.to_csv(file_path)


def getting_files(filename):
    file_path = 'C:/Users/Ankush/PycharmProjects/Django/auth_app/files/' + filename
    # absolute_path = os.path.abspath(file_path)
    # print(f"Absolute path: {absolute_path}")
    # print(f"File exists: {os.path.exists(absolute_path)}")
    df = pd.read_csv(file_path, index_col=0)
    df.reset_index(inplace=True)
    return df


def get_daily_time_seriess(function, symbol):
    # Example base URL for Alpha Vantage
    base_url = "https://www.alphavantage.co/query"
    api_key = "W218WSA3VU5SDWJQ"  # Make sure to use your actual API key

    # Construct the full API request URL
    url = f"{base_url}?function={function[0]}&symbol={symbol}&apikey={api_key}&outputsize=full"

    # Fetch the data from the API
    response = requests.get(url)
    # Check for the rate limit message in the response
    data = response.json()
    if "Information" in data:
        return None, data["Information"]  # Return None for data and the message

    return data, None
    # return response.json()
