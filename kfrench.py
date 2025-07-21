import os
import shutil
import zipfile

import numpy as np
import pandas as pd
import requests

import utils

# Define URLs for downloading data
INDUSTRY_PORTFOLIOS_URL = (
    'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/48_Industry_Portfolios_daily_CSV.zip'
)
FRENCH_DATA_FACTORS_URL = (
    'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
)
FRENCH_DATA_FACTORS_LOCAL_PATH = utils.ROOT_DIR / 'data' / 'french_factors_daily.csv'
TEMP_FOLDER = 'temp_folder'


def download_and_extract_zip(url, extract_to=TEMP_FOLDER):
    """
    Downloads a ZIP file from a URL and extracts its contents.

    Args:
    url (str): The URL to download the ZIP file from.
    extract_to (str): The folder to extract the contents to.

    Returns:
    str: Path to the first CSV file found in the extracted contents.
    """
    zip_file = 'temp.zip'
    os.makedirs(extract_to, exist_ok=True)

    # Download the file
    with open(zip_file, 'wb') as f:
        f.write(requests.get(url).content)

    # Extract the ZIP file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    os.remove(zip_file)

    # Get the path to the extracted CSV file
    extracted_files = os.listdir(extract_to)
    csv_files = [f for f in extracted_files if f.lower().endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError('No CSV file found in the extracted folder.')

    return os.path.join(extract_to, csv_files[0])


def cleanup_temp_files(folder=TEMP_FOLDER):
    """
    Removes temporary files and folder.

    Args:
    folder (str): The folder to clean up.
    """
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))
    os.rmdir(folder)


def process_csv_file(csv_file_path):
    """
    Reads and processes the CSV file.

    Args:
    csv_file_path (str): Path to the CSV file.

    Returns:
    DataFrame: Processed data.
    DatetimeIndex: Dates.
    """
    csv_data = pd.read_csv(csv_file_path, skiprows=9, low_memory=False)
    csv_data = csv_data.apply(pd.to_numeric, errors='coerce')

    industry_names = csv_data.columns[1:]
    rows_nan = csv_data[csv_data.iloc[:, 0].isna()].index
    from_row = 0
    until_row = rows_nan[0]
    data = csv_data.iloc[from_row:until_row, :].to_numpy()

    ret = data[:, 1:] / 100
    caldt = pd.to_datetime(data[:, 0].astype(int).astype(str), format='%Y%m%d')
    ret[ret <= -0.99] = np.nan

    return pd.DataFrame(data=ret, index=caldt, columns=industry_names), caldt


def market_french_reconciled(caldt):
    """
    Reconciles market returns using Fama-French data.

    Args:
    caldt (DatetimeIndex): Dates.

    Returns:
    Series: Market returns.
    """
    csv_file_path = download_and_extract_zip(FRENCH_DATA_FACTORS_URL)
    csv_data = pd.read_csv(csv_file_path, skiprows=3, header=None, low_memory=False)

    # Coerce non-numeric data to NaN and then drop rows with NaN in the date column
    csv_data = csv_data.apply(pd.to_numeric, errors='coerce')
    csv_data = csv_data.dropna(subset=[0])  # Drop rows where the date column is NaN

    # Convert the first column to datetime
    caldt_mkt = pd.to_datetime(csv_data.iloc[:, 0].astype(int).astype(str), format='%Y%m%d')

    # Calculate market returns by adding Mkt-RF and RF and dividing by 100 to convert to decimal form
    ret_mkt = (csv_data.iloc[:, 1] + csv_data.iloc[:, 4]) / 100

    # Create a series filled with NaN values for the length of 'caldt'
    y = np.full(len(caldt), np.nan)

    # Find indices where dates match and assign values accordingly
    idx = np.where(np.in1d(caldt_mkt, caldt))[0]
    y[np.in1d(caldt, caldt_mkt)] = ret_mkt.iloc[idx]

    cleanup_temp_files()

    return pd.Series(data=y, index=caldt)


def tbill_french_reconciled(caldt):
    """
    Reconciles T-Bill returns using Fama-French data.

    Args:
    caldt (DatetimeIndex): Dates.

    Returns:
    Series: T-Bill returns.
    """
    already_downloaded = False
    if not FRENCH_DATA_FACTORS_LOCAL_PATH.exists():
        csv_file_path = download_and_extract_zip(FRENCH_DATA_FACTORS_URL)
        shutil.copy(csv_file_path, FRENCH_DATA_FACTORS_LOCAL_PATH)
    else:
        already_downloaded = True
    csv_data = pd.read_csv(FRENCH_DATA_FACTORS_LOCAL_PATH, skiprows=3, header=None, low_memory=False)

    # Coerce non-numeric data to NaN and then drop rows with NaN in the date column
    csv_data = csv_data.apply(pd.to_numeric, errors='coerce')
    csv_data = csv_data.dropna(subset=[0])  # Drop rows where the date column is NaN

    # Convert the first column to datetime
    caldt_tbill = pd.to_datetime(csv_data.iloc[:, 0].astype(int).astype(str), format='%Y%m%d')

    # Extract T-Bill returns and divide by 100 to convert to decimal form
    ret_tbill = csv_data.iloc[:, 4] / 100

    # Create a series filled with NaN values for the length of 'caldt'
    y = np.full(len(caldt), np.nan)

    # Find indices where dates match and assign values accordingly
    idx = np.where(np.in1d(caldt_tbill, caldt))[0]
    y[np.in1d(caldt, caldt_tbill)] = ret_tbill.iloc[idx]

    if not already_downloaded:
        cleanup_temp_files()

    return pd.Series(data=y, index=caldt)