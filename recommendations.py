from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
# Autres imports nécessaires (par exemple, pour le traitement des données et la comparaison des embeddings)
# Exemple :
# from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Configuration de la connexion à MongoDB
client = MongoClient('mongodb://localhost:27017/', replicaSet='<nom_du_replica_set>') # Remplacez <nom_du_replica_set> par le nom de votre replica set
db = client.nom_de_la_base_de_donnees # Remplacez nom_de_la_base_de_donnees par le nom de votre base de données

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    """
    Recommande des films basés sur un embedding donné.
    """
    try:
        # 1. Récupérer l'embedding depuis la requête POST
        data = request.get_json()
        target_embedding = data.get('embedding')

        # Vérifier si l'embedding est fourni
        if not target_embedding:
            return jsonify({'error': 'Embedding is required'}), 400

        target_embedding = np.array(target_embedding) # Convertir la liste en un tableau numpy

        # 2. Récupérer tous les films depuis MongoDB
        movies = list(db.movies.find({}, {'_id': 1, 'embedding': 1}))

        # 3. Comparer l'embedding cible avec les embeddings des films
        # (Vous pouvez utiliser la similarité cosinus ou une autre métrique)
        # similarities = cosine_similarity([target_embedding], [movie['embedding'] for movie in movies])
        # Exemple (remplacer par votre implémentation)
        similarities = [np.dot(target_embedding, movie['embedding']) for movie in movies] # Produit scalaire pour l'exemple

        # 4. Récupérer les films les plus similaires (recommandations)
        # (Trier les films par similarité et récupérer les plus pertinents)
        similar_movies_indices = np.argsort(similarities)[::-1]  # Indices triés par similarité décroissante
        recommended_movies = [movies[i] for i in similar_movies_indices]

        # 5. Récupérer les informations complètes des films recommandés depuis MongoDB
        # (Vous pouvez récupérer les informations complètes en utilisant les IDs des films recommandés)
        # Exemple :
        results = []
        for movie in recommended_movies:
            full_movie_data = db.movies.find_one({'_id': movie['_id']})
            results.append(full_movie_data)

        return jsonify({'recommendations': results}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
