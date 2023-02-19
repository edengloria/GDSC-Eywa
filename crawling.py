import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Prompt the user to enter the search keyword and the number of images to download
keyword = input('Enter the search keyword: ')
num_images = int(input('Enter the number of images to download: '))

# URL of the image search page for the given keyword
url = f'https://www.google.com/search?q={keyword}&tbm=isch'

# Request the search page and parse the HTML response
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all image tags and extract the image source URLs
image_tags = soup.find_all('img')
image_urls = [tag['src'] for tag in image_tags]

# Create the folder to save the images if it doesn't exist
folder_path = f'Eywa_dataset/{keyword}'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Download and save the specified number of images
for i, url in enumerate(image_urls):
    if i == num_images:
        break
    try:
        response = requests.get(url)
        file_path = os.path.join(folder_path, f'{keyword}_{i}.jpg')
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f'Saved {file_path}')
    except:
        print(f'Error saving {url}')
