from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
# Autres imports nécessaires (par exemple, pour le traitement des images et la comparaison des embeddings)
# Exemple :
# from PIL import Image
# from io import BytesIO
# from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Configuration de la connexion à MongoDB
client = MongoClient('mongodb://localhost:27017/', replicaSet='<nom_du_replica_set>') # Remplacez <nom_du_replica_set> par le nom de votre replica set
db = client.nom_de_la_base_de_donnees # Remplacez nom_de_la_base_de_donnees par le nom de votre base de données

@app.route('/search_by_poster', methods=['POST'])
def search_by_poster():
    """
    Recherche des films similaires à une image donnée en comparant les embeddings.
    """
    try:
        # 1. Récupérer l'image depuis la requête POST
        image_file = request.files['image']
        # Exemple de traitement de l'image (à adapter selon votre méthode)
        # image = Image.open(BytesIO(image_file.read()))

        # 2. Générer l'embedding de l'image avec le modèle ResNet
        # (Vous devrez implémenter cette partie en utilisant votre modèle ResNet)
        # query_embedding = generate_embedding(image)
        # Exemple (remplacer par votre implémentation)
        query_embedding = np.random.rand(512)  # Vecteur aléatoire pour l'exemple

        # 3. Récupérer tous les embeddings des films depuis MongoDB
        movies = list(db.movies.find({}, {'_id': 1, 'embedding': 1}))

        # 4. Comparer l'embedding de l'image avec les embeddings des films
        # (Vous pouvez utiliser la similarité cosinus ou une autre métrique)
        # similarities = cosine_similarity([query_embedding], [movie['embedding'] for movie in movies])
        # Exemple (remplacer par votre implémentation)
        similarities = [np.dot(query_embedding, movie['embedding']) for movie in movies] # Produit scalaire pour l'exemple

        # 5. Récupérer les films les plus similaires
        # (Trier les films par similarité et récupérer les plus pertinents)
        similar_movies_indices = np.argsort(similarities)[::-1]  # Indices triés par similarité décroissante
        similar_movies = [movies[i] for i in similar_movies_indices]

        # 6. Récupérer les informations complètes des films similaires depuis MongoDB
        # (Vous pouvez récupérer les informations complètes en utilisant les IDs des films similaires)
        # Exemple :
        results = []
        for movie in similar_movies:
            full_movie_data = db.movies.find_one({'_id': movie['_id']})
            results.append(full_movie_data)

        return jsonify({'similar_movies': results}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
