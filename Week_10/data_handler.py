import requests


def fetch_iris(path):
    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/iris.scale"

    response = requests.get(url)

    if response.status_code == 200:
        with open(path, "wb") as file:
            file.write(response.content)
        print(f"File downloaded and saved as {path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
