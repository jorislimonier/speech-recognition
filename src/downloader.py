# %%
import urllib

import requests
from bs4 import BeautifulSoup

# %%
class Downloader:
    def __init__(self) -> None:
        pass

    BASE_URL = "https://catalog.ldc.upenn.edu/docs/LDC97S62/"

    def download(self, filename=None):
        req = requests.get(url=self.BASE_URL)
        soup = BeautifulSoup(req.content, "html.parser")

        files = [tr.td.a.get("href") for tr in soup.tbody.find_all("tr")][1:]
        for file in files:
            response = urllib.request.urlretrieve(
                url=self.BASE_URL + file,
                filename=f"../data/raw/{file}",
            )
downloader = Downloader()
downloader.download()