import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import requests
from bs4 import BeautifulSoup


# Get hero names list
hero_names_path = "./test_data/hero_names.txt"
with open(hero_names_path, "r") as f:
    hero_names = f.read().splitlines()


def get_champion_hero_image_links(hero_names):
    url = "https://leagueoflegends.fandom.com/wiki/Champion_(Wild_Rift)"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    hero_image_links = {}

    for hero_name in tqdm(hero_names):
        
        # Replace spaces with underscores and fix Kai'Sa and Kha'Zix
        new_hero_name = hero_name.replace("_", " ")
        new_hero_name = "Kai'Sa" if new_hero_name == "KaiSa" else new_hero_name
        new_hero_name = "Kha'Zix" if new_hero_name == "KhaZix" else new_hero_name

        link = soup.find_all("img", attrs={'alt': new_hero_name})[0]
        
        match = re.search(r'https://.*?\.png', link["src"])

        if match:
            extracted_link = match.group(0)
        else:
            match = re.search(r'https://.*?\.png', link["data-src"])
            if match:
                extracted_link = match.group(0)
            else:
                print("No match found for ", hero_name)
                continue

        hero_image_links[hero_name] = extracted_link

    return hero_image_links


def download_hero_images(hero_image_links, path):
    for hero_name, link in tqdm(hero_image_links.items()):
        response = requests.get(link)
        
        if response.status_code == 200:
            with open(f"{path}{hero_name}.png", "wb") as f:
                f.write(response.content)
        else:
            print(f"Failed to download image for {hero_name}.")

# download_hero_images(hero_image_links, path="./test_data/hero_images/")