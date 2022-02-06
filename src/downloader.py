# %%
import os
import shutil
import tarfile
import urllib

import requests
from bs4 import BeautifulSoup


class Downloader:
    def __init__(self) -> None:
        pass

    BASE_URL = "https://catalog.ldc.upenn.edu/docs/LDC97S62/"
    RAW_PATH = f"./data/raw/"

    def download(self):
        """
        Downloads all files in `RAW_PATH`. \\
        If a file ends with ".tar.gz", creates a directory with its name (without ".tar.gz"), 
        moves it there and extracts it.
        """
        # Get html soup from `BASE_URL` link
        req = requests.get(url=self.BASE_URL)
        soup = BeautifulSoup(req.content, "html.parser")

        # Extract URLs
        files = [tr.td.a.get("href") for tr in soup.tbody.find_all("tr")][1:]

        for file in files:
            filepath = f"{self.RAW_PATH}{file}"
            
            # Continue if file/dir already exists
            if os.path.exists(filepath.removesuffix('.tar.gz')):
                continue

            # Download `file`
            urllib.request.urlretrieve(
                url=self.BASE_URL + file,
                filename=filepath,
            )

            # If tar file, move to its own directory and extract it
            if file.endswith(".tar.gz"):
                folder_from_file = f"{filepath.removesuffix('.tar.gz')}/"
                
                # Create directory from filename without extenstion
                # if not os.path.exists(folder_from_file):
                os.mkdir(path=folder_from_file)
                if os.path.exists(f"{folder_from_file}{file}"):
                    print("File already exists")
                    os.remove(filepath)
                else:
                    shutil.move(
                        src=filepath,
                        dst=folder_from_file,
                    )
                with tarfile.open(f"{folder_from_file}{file}") as f:
                    f.extractall(folder_from_file)


downloader = Downloader()
downloader.download()

