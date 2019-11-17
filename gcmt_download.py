import requests
import os
import bs4

# Download new monthly files (2005 - 2019)
path = './gcmt_data/'
catalog_url = "http://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY/"

Year = 2005

while True:
    url = catalog_url + str(Year) + '/'

    r = requests.get(url)
    data = bs4.BeautifulSoup(r.text, "html.parser")
    for l in data.find_all("a"):
        r = requests.get(url + l["href"])
        if os.path.exists(path + l["href"]):
            print(l["href"], 'already exists')
            continue
        try:
            with open(path + l["href"], "wb") as file:
                file.write(r.content)
                print('Downloaded', l["href"])
        except:
            continue

    Year += 1
