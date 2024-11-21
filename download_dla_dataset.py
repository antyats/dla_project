import argparse
import urllib
from pathlib import Path
from zipfile import ZipFile

import requests

YANDEX_API_URL = "https://cloud-api.yandex.net/v1/disk/public/resources/download"


def get_download_url(public_url):
    response = requests.get(YANDEX_API_URL, params={"public_key": public_url})
    if response.status_code == 200:
        return response.json().get("href")
    else:
        raise Exception("Failed to retrieve download URL")


def download_dataset(download_url, download_location, extract):
    download_location.mkdir(parents=True, exist_ok=True)
    file_name = urllib.parse.unquote(download_url.split("filename=")[1].split("&")[0])
    save_path = download_location / file_name

    print("Downloading...")
    with save_path.open("wb") as file:
        download_response = requests.get(download_url, stream=True)
        for chunk in download_response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                file.flush()
    print(f"Downloaded to {save_path}")

    if extract:
        print("Extracting...")
        with ZipFile(save_path, "r") as zf:
            zf.extractall(download_location)
        print(f"Extracted to {download_location}")


def main():
    parser = argparse.ArgumentParser(
        description="Download dla dataset from Yandex Disk."
    )
    parser.add_argument("url", type=str, help="Link to the file on Yandex Disk")
    parser.add_argument("dir", type=Path, help="The folder to download dataset to")
    parser.add_argument(
        "--extract", action="store_true", help="Extract the zip-file after download"
    )

    args = parser.parse_args()
    download_url = get_download_url(args.url)

    download_dataset(download_url, args.dir, args.extract)


if __name__ == "__main__":
    main()
