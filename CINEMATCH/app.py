from flask import Flask, render_template, request, jsonify, redirect
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId
import io
from base64 import b64encode
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# --- Connexion à MongoDB ---
try:
    # Pour se connecter au cluster sharded avec réplica, ajoutez l'option replicaSet si nécessaire
    client = MongoClient("mongodb://localhost:27017/")
    db = client["movie_database"]
    movies_collection = db["movies"]
    fs = gridfs.GridFS(db)
    print("Connected to MongoDB")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

# --- Initialisation du modèle ResNet pour générer les embeddings ---
model = models.resnet18(pretrained=True)
model.eval()
embedding_model = torch.nn.Sequential(*list(model.children())[:-1])
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def generate_embedding(image_bytes):
    """Génère un embedding à partir des bytes d'une image."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            embedding = embedding_model(input_tensor)
        embedding = embedding.squeeze().numpy().flatten()
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def cosine_similarity(a, b):
    """Calcule la similarité cosinus entre deux vecteurs."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

@app.route('/')
def index():
    movies = list(movies_collection.find())
    movie_list = []
    for movie in movies:
        poster_data = None
        if movie.get("image_id"):
            try:
                grid_out = fs.get(ObjectId(movie["image_id"]))
                poster_data = b64encode(grid_out.read()).decode('utf-8')
            except Exception as e:
                print(f"Error getting poster for {movie.get('title', 'No title')}: {e}")
        movie_list.append({
            "title": movie.get("title", "No title"),
            "overview": movie.get("overview", "No overview available"),
            "poster_data": poster_data,
        })
    return render_template('index.html', movies=movie_list)

@app.route('/movie/<int:movie_index>')
def movie_detail(movie_index):
    movies = list(movies_collection.find())
    if 0 <= movie_index < len(movies):
        movie = movies[movie_index]
        poster_data = None
        if movie.get("image_id"):
            try:
                grid_out = fs.get(ObjectId(movie["image_id"]))
                poster_data = b64encode(grid_out.read()).decode('utf-8')
            except Exception as e:
                print(f"Error getting poster for {movie.get('title', 'No title')}: {e}")
        return render_template('movie_detail.html', movie={
            "title": movie.get("title", "No title"),
            "overview": movie.get("overview", "No overview available"),
            "poster_data": poster_data,
            "release_date": movie.get("release_date", "N/A"),
            "runtime": movie.get("runtime", "N/A"),
            "status": movie.get("status", "N/A"),
            "original_language": movie.get("original_language", "N/A"),
            "budget": movie.get("budget", "N/A"),
            "revenue": movie.get("revenue", "N/A"),
            "genres": movie.get("genres", [])
        })
    else:
        return "Movie not found", 404

@app.route('/search_by_poster', methods=['POST'])
def search_by_poster():
    """
    Reçoit une image via une requête POST, calcule son embedding,
    puis compare avec les embeddings de la collection pour trouver les films similaires.
    """
    if 'poster' not in request.files:
        return jsonify({"error": "No poster file provided"}), 400
    file = request.files['poster']
    image_bytes = file.read()
    query_embedding = generate_embedding(image_bytes)
    if query_embedding is None:
        return jsonify({"error": "Error generating embedding"}), 500

    # On récupère tous les films qui ont un embedding stocké
    movies = list(movies_collection.find({"embedding": {"$ne": None}}))
    results = []
    for movie in movies:
        emb = movie.get("embedding")
        if emb is not None:
            sim = cosine_similarity(np.array(query_embedding), np.array(emb))
            results.append((sim, movie))
    
    # Trier par similarité décroissante et retourner par exemple les 5 premiers résultats
    results.sort(key=lambda x: x[0], reverse=True)
    
    # Modification ici pour inclure les données du poster
    top_results = [
        {
            "title": m.get("title", "No title"), 
            "similarity": s,
            "poster_data": m.get("poster_data", ""),  # Ajout des données de poster
            "overview": m.get("overview", "")  # Optionnel, mais utile
        } for s, m in results[:5]
    ]
    
    return jsonify(top_results)

# Vous pouvez ajouter d'autres routes (/movies, /scrape_movies, /recommendations, /stats, etc.) selon les besoins.


# Ajoutez ces routes à votre fichier app.py
@app.route('/delete_movie/<int:movie_index>', methods=['DELETE'])
def delete_movie(movie_index):
    try:
        movies = list(movies_collection.find())
        if 0 <= movie_index < len(movies):
            movie = movies[movie_index]
            
            # Supprimer l'image GridFS si elle existe
            if movie.get("image_id"):
                fs.delete(ObjectId(movie["image_id"]))
            
            # Supprimer le document de la collection
            movies_collection.delete_one({"_id": movie["_id"]})
            
            return jsonify({"message": "Film supprimé avec succès"}), 200
        else:
            return jsonify({"error": "Film non trouvé"}), 404
    except Exception as e:
        print(f"Erreur lors de la suppression: {e}")
        return jsonify({"error": "Erreur lors de la suppression"}), 500

@app.route('/edit_movie/<int:movie_index>', methods=['GET', 'POST'])
def edit_movie(movie_index):
    movies = list(movies_collection.find())
    if 0 <= movie_index < len(movies):
        movie = movies[movie_index]
        
        if request.method == 'GET':
            # Récupérer l'image pour l'affichage
            poster_data = None
            if movie.get("image_id"):
                try:
                    grid_out = fs.get(ObjectId(movie["image_id"]))
                    poster_data = b64encode(grid_out.read()).decode('utf-8')
                except Exception as e:
                    print(f"Erreur lors de la récupération de l'image: {e}")
            
            movie['poster_data'] = poster_data
            return render_template('edit_movie.html', movie=movie)
        
        elif request.method == 'POST':
            # Identifier la clé de sharding
            shard_key = None
            try:
                # Essayez de récupérer la configuration de la collection
                config_db = client.config
                collections = config_db.collections
                collection_config = collections.find_one({"_id": f"{db.name}.{movies_collection.name}"})
                
                if collection_config and 'key' in collection_config:
                    shard_key = list(collection_config['key'].keys())[0]
                    print(f"Clé de sharding identifiée : {shard_key}")
                else:
                    print("Impossible de trouver la clé de sharding")
            except Exception as e:
                print(f"Erreur lors de la récupération de la clé de sharding : {e}")
            
            # Préparer les données de mise à jour
            update_data = {
                "title": request.form.get("title"),
                "overview": request.form.get("overview"),
                "release_date": request.form.get("release_date")
            }
            
            # Gestion de la nouvelle affiche
            poster = request.files.get('poster')
            if poster and poster.filename:
                # Supprimer l'ancienne image si elle existe
                if movie.get("image_id"):
                    fs.delete(ObjectId(movie["image_id"]))
                
                # Stocker la nouvelle image
                poster_bytes = poster.read()
                image_id = fs.put(poster_bytes, filename=secure_filename(poster.filename))
                update_data["image_id"] = str(image_id)
                
                # Régénérer l'embedding
                embedding = generate_embedding(poster_bytes)
                if embedding is not None:
                    update_data["embedding"] = embedding.tolist()
            
            # Préparer la requête de mise à jour
            update_query = {"_id": movie["_id"]}
            
            # Si une clé de sharding est trouvée, l'ajouter à la requête
            if shard_key and shard_key != '_id':
                update_query[shard_key] = movie.get(shard_key)
            
            try:
                # Tenter plusieurs méthodes de mise à jour
                result = movies_collection.replace_one(
                    update_query, 
                    {**movie, **update_data}
                )
                
                if result.modified_count == 0:
                    # Fallback method
                    movies_collection.update_one(
                        update_query, 
                        {"$set": update_data},
                        bypass_document_validation=True
                    )
                
                print("Mise à jour réussie")
                return redirect('/')
            
            except Exception as e:
                print(f"Erreur de mise à jour complète : {e}")
                
                # Dernière tentative avec une mise à jour minimale
                try:
                    movies_collection.update_one(
                        {"_id": movie["_id"]}, 
                        {"$set": update_data}
                    )
                    return redirect('/')
                except Exception as final_error:
                    print(f"Erreur finale : {final_error}")
                    return f"Erreur de mise à jour : {final_error}", 500
    
    return "Film non trouvé", 404




if __name__ == '__main__':
    app.run(debug=True)
