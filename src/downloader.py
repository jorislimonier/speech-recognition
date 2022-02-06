# %%
import urllib

import requests
from bs4 import BeautifulSoup

# %%
BASE_URL = "https://catalog.ldc.upenn.edu/docs/LDC97S62/"
req = requests.get(url=BASE_URL)
soup = BeautifulSoup(req.content, "html.parser")

files = [tr.td.a.get("href") for tr in soup.tbody.find_all("tr")][1:]
files
# %%
# %%
response = urllib.request.urlretrieve(BASE_URL + files[1], f"data/{files[1]}")
print(response)
