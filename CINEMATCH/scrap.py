import gridfs
from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
import time
import datetime
import io

# Pour la génération d'embeddings
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# --- Initialisation du modèle ResNet pour générer les embeddings ---
# Utilisons resnet18 dont la couche finale (après pooling) produit un vecteur de 512 dimensions.
model = models.resnet18(pretrained=True)
model.eval()  # Mode évaluation
# On retire la dernière couche fully-connected pour récupérer le vecteur avant classification
embedding_model = torch.nn.Sequential(*list(model.children())[:-1])
# Définir la transformation pour les images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def generate_embedding(image_bytes):
    """Génère un embedding (liste de 512 floats) à partir des bytes d'une image."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)  # Ajouter la dimension batch
        with torch.no_grad():
            embedding = embedding_model(input_tensor)
        # Le vecteur de sortie est de taille [1, 512, 1, 1] que nous aplatissons en 1D
        embedding = embedding.squeeze().numpy().flatten().tolist()
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def extract_release_year(release_date_str):
    """Extrait l'année de la date de sortie au format 'YYYY-MM-DD'."""
    try:
        dt = datetime.datetime.strptime(release_date_str, '%Y-%m-%d')
        return dt.year
    except Exception as e:
        print(f"Error extracting year from {release_date_str}: {e}")
        return None

# --- Connexion MongoDB ---
try:
    # Ici, adaptez la chaine de connexion pour vous connecter à votre cluster sharded
    client = MongoClient("mongodb://localhost:27017/")  
    db = client["movie_database"]  # Nom de la base
    movies_collection = db["movies"]
    fs = gridfs.GridFS(db)
    print("Connected to MongoDB")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

def insert_movies_to_mongodb(movie_data_list, collection, fs):
    """Insère une liste de films dans la collection MongoDB en enregistrant le poster dans GridFS,
    en générant l'embedding et en ajoutant release_year."""
    if not movie_data_list:
        print("No movie data to insert.")
        return

    movies_to_insert = []
    for movie in movie_data_list:
        # Téléchargement et enregistrement du poster via GridFS
        if movie.get("poster"):
            try:
                response = requests.get(movie["poster"])
                response.raise_for_status()
                poster_bytes = response.content
                poster_id = fs.put(poster_bytes, filename=f"{movie['title']}_poster")
                movie["image_id"] = poster_id  # Nom plus explicite
                # Génération de l'embedding du poster
                embedding = generate_embedding(poster_bytes)
                movie["embedding"] = embedding
                del movie["poster"]  # On ne stocke plus l'URL
            except requests.exceptions.RequestException as e:
                print(f"Error downloading poster for {movie['title']}: {e}")
                movie["image_id"] = None
                movie["embedding"] = None
        else:
            movie["image_id"] = None
            movie["embedding"] = None

        # Ajout d'un champ release_year pour servir de clé de sharding (par exemple)
        if movie.get("release_date"):
            movie["release_year"] = extract_release_year(movie["release_date"])
        else:
            movie["release_year"] = None

        # Vous pouvez restructurer ou ajouter d'autres champs pour coller au modèle demandé (ex: additional_info, cast, etc.)
        movies_to_insert.append(movie)

    try:
        result = collection.insert_many(movies_to_insert)
        print(f"Inserted {len(result.inserted_ids)} movies into MongoDB.")
    except Exception as e:
        print(f"Error inserting movies into MongoDB: {e}")

def scrape_movie_details(movie_url):
    """Scrape les détails d'un film à partir d'une page TMDb."""
    try:
        response = requests.get(movie_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        details = {}
        # Exemple pour récupérer le poster et d'autres infos
        poster_tag = soup.find("img", class_="poster")
        if poster_tag and poster_tag.get("src"):
            details["poster"] = poster_tag["src"]
        title_element = soup.find("div", class_="title")
        if title_element:
            details["title"] = title_element.find("a").get_text(strip=True)
            # Exemple : extraire l'année depuis un élément entre parenthèses
            year_text = title_element.find("span").get_text(strip=True).replace("(", "").replace(")", "")
            # On suppose ici que la date de sortie se présente sous forme YYYY-MM-DD
            details["release_date"] = f"{year_text}-01-01"  # Adaptation en cas de format différent
        overview_element = soup.find("div", class_="overview")
        if overview_element:
            details["overview"] = overview_element.get_text(strip=True)
        # Vous pouvez extraire d'autres informations (runtime, status, genres, cast, etc.)
        release_date_element = soup.find("span", class_="release")
        if release_date_element:
            details["release_date"] = release_date_element.get_text(strip=True)
        genres_element = soup.find("span", class_="genres")
        if genres_element:
            details["genres"] = [a.get_text(strip=True) for a in genres_element.find_all("a")]
        # Exemple pour récupérer un casting basique
        cast_section = soup.find("ol", class_="people")
        if cast_section:
            character_elements = cast_section.find_all("li")
            details["cast"] = [
                {
                    "name": character.find("a").get_text(strip=True),
                    "role": character.find("p", class_="character").get_text(strip=True)
                }
                for character in character_elements if character.find("a") and character.find("p", class_="character")
            ]
        # Autres champs supplémentaires (budget, revenue, etc.) peuvent être extraits de la même manière.
        return details

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    except AttributeError as e:
        print(f"Error parsing HTML: {e}")
        return None

def scrape_movie_links(base_url, pages_to_scrape=1):
    """Scrape les liens des films sur plusieurs pages d'une liste TMDb."""
    movie_links = []
    for page_num in range(1, pages_to_scrape + 1):
        url = f"{base_url}?page={page_num}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            movie_cards = soup.find_all("div", class_="card style_1")
            for movie_card in movie_cards:
                movie_element = movie_card.find("a")
                link = movie_element["href"]
                full_link = f"https://www.themoviedb.org{link}"
                movie_links.append(full_link)
            time.sleep(1)  # Pause pour ne pas surcharger le serveur
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page_num}: {e}")
    return movie_links

# --- Exemple d'utilisation ---
base_url = "https://www.themoviedb.org/movie"
movie_links = scrape_movie_links(base_url, pages_to_scrape=1)
movie_data_list = []
for movie_link in movie_links:
    movie_data = scrape_movie_details(movie_link)
    if movie_data:
        movie_data_list.append(movie_data)
        print(f"Scraped: {movie_data.get('title', 'No title')}")
    time.sleep(1)

# Afficher les films scrappés (pour vérification)
for movie in movie_data_list:
    print(movie)

insert_movies_to_mongodb(movie_data_list, movies_collection, fs)
client.close()
