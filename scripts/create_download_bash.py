import numpy as np
import requests
import pandas as pd
from collections import defaultdict

class TableExtractor:
    """
    A class to extract tables from a list of URLs stored in a text file.
    """

    def __init__(self, url_file):
        """
        Initializes the TableExtractor with the path to the URL file.

        :param url_file: The path to the .txt file containing URLs.
        """
        self.url_file = url_file
        self.urls = self._read_urls_from_file()

    def _read_urls_from_file(self):
        """
        Reads URLs from the provided text file and returns them as a list.
        This is a private helper method for the constructor.
        """
        try:
            with open(self.url_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            return urls
        except FileNotFoundError:
            print(f"Error: The file '{self.url_file}' was not found.")
            return []

    def _get_html_content(self, url):
        """
        Fetches the HTML content of a given URL.
        This is a private helper method.
        """
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Could not fetch content from {url}. Error: {e}")
            return None

    def extract_tables_from_url(self, url):
        """
        Extracts all tables from a given URL's HTML content.

        :param url: The URL to extract tables from.
        :return: A list of pandas DataFrames, where each DataFrame is a table.
        """
        html_content = self._get_html_content(url)
        if html_content:
            try:
                # pandas.read_html is powerful and finds all <table> tags
                tables = pd.read_html(html_content, converters=defaultdict(lambda: str))
                return tables
            except ValueError:
                # This error is raised by pandas if no tables are found
                print(f"Info: No tables were found on {url}.")
                return []
        return []

    def process_all_urls(self):
        """
        A generator that processes each URL from the file and yields the URL
        and a list of its extracted tables.
        """
        if not self.urls:
            print("URL list is empty. Cannot process.")
            return

        for url in self.urls:
            tables = self.extract_tables_from_url(url)
            yield url, tables

def main():
    url_file_path = 'download_links.txt'
    
    extractor = TableExtractor(url_file_path)
    
    download_comamnd_list = []
    i = 0
    for url, list_of_tables in extractor.process_all_urls():
        i = i+1

        if (i%2 == 1):
            continue

        table_df = list_of_tables[0]

        for j,row in table_df.iterrows():
            noaa_ar = row['NOAA AR']
            date_of_observation = row['Date of observation']
            time_of_observation = row['Time of observation']
            download_command = f"netcdf_files+=('v12/{noaa_ar}/{date_of_observation}/{noaa_ar}_{date_of_observation}_{time_of_observation}.nc')"
            download_comamnd_list.append(download_command)

            print(download_command)

        
    
    download_comamnd_list_with_newlines = [item + "\n" for item in download_comamnd_list]

    with open("full_download_list.tsx", "w") as f:
        f.writelines(download_comamnd_list_with_newlines)


if __name__ == "__main__":
    with open("NLFFF Data/full_download_list.txt", "r") as f:
        lines = [line.rstrip() for line in f]

    download_links = np.array(lines)
    np.random.shuffle(download_links)
    download_links = download_links[:400]
    
    download_links.tofile('small_download.sh', sep='\n')