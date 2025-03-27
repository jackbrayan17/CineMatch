# CINEMATCH

## Movie Poster Similarity Search Application

### Overview

This Flask-based web application allows users to explore a movie database and search for similar movies based on movie posters. The app leverages advanced image recognition techniques, using pre-trained deep learning models, specifically ResNet, to generate image embeddings that are used to calculate the similarity between different movie posters. The system then provides a list of the most similar movies based on the poster image uploaded by the user.

## Key Features

- **Movie Display:** A simple, intuitive interface that displays movies along with their posters and overviews.
- **Poster-based Search:** Users can upload a movie poster image, and the app will find movies in the database with similar posters.
- **Database Management:** Manage the movie collection by adding, editing, and deleting movies along with their images.
- **Machine Learning Integration:** Utilizes a pre-trained ResNet-18 deep learning model to generate embeddings from movie posters for similarity comparison.
- **MongoDB Backend:** The app stores movie data and images in MongoDB, leveraging GridFS for storing large images and embeddings for fast retrieval and search.

## Functionalities

### 1. **Movie Overview**
The homepage displays a list of movies with the following details:
- Title
- Overview
- Poster image

### 2. **Movie Detail Page**
Clicking on a movie title will take you to a detailed page for that specific movie, which includes:
- Title
- Overview
- Poster image
- Release date, runtime, budget, revenue, and genres.

### 3. **Search by Movie Poster**
Users can upload a movie poster, and the app will generate an embedding for the image. It will then compare this embedding to the embeddings of all movies in the database and return the most similar results based on cosine similarity. The top 5 similar movies are displayed along with their titles, overviews, and posters.

### 4. **Movie CRUD Operations**
- **Create (Add New Movie):** Add new movies to the database by submitting a title, overview, release date, and poster.
- **Read (View Movie Details):** View movie details along with poster images and other related data.
- **Update (Edit Movie):** Edit movie information and update movie posters and embeddings.
- **Delete (Remove Movie):** Remove movies from the database and delete associated poster images from GridFS.

## Technical Details

### Backend
- **Flask:** Web framework used to create the app.
- **MongoDB:** NoSQL database for storing movie data and images.
- **GridFS:** Used for storing large movie poster images.
- **Torch & torchvision:** Used to implement the image embedding generation process using ResNet-18 model.
- **PIL (Pillow):** Used to process images before generating embeddings.
  
### Deep Learning Model
The app uses a pre-trained ResNet-18 model from the `torchvision` library to generate embeddings from movie poster images. These embeddings are then used to calculate cosine similarity between the uploaded poster and those stored in the database.

### Embedding Generation
- The poster image is resized and normalized before being passed through the model.
- The output embedding is then compared with other embeddings in the database to find the most similar images.

### Cosine Similarity
The similarity between the uploaded poster's embedding and those stored in the database is computed using the cosine similarity metric. The result is a ranking of movies based on how similar their posters are.

## How to Run

1. Clone the repository to your local machine:
   ```
   git clone https://github.com/your-username/movie-poster-similarity.git
   cd movie-poster-similarity
   ```

2. Set up a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Ensure MongoDB is installed and running on your local machine (or configure the MongoDB URI to point to a remote server).

5. Run the Flask app:
   ```
   python app.py
   ```

6. Access the app in your browser at `http://127.0.0.1:5000/`.

## Innovation

This application innovatively integrates machine learning into a web application to create a more engaging movie exploration experience. By allowing users to find similar movies based on poster images, the app opens up new ways of discovering films. Instead of relying on tags or descriptions, users can search for movies based on visual similarity, leveraging the power of deep learning and computer vision.

## Technologies Used

- **Flask** for building the web application.
- **MongoDB** for storing movie data and images.
- **PyTorch** for deep learning-based image processing.
- **ResNet-18** for generating image embeddings.
- **GridFS** for managing large files like movie posters.
- **HTML/CSS/JavaScript** for front-end rendering.

## Future Enhancements

- **Search by Movie Genre:** Implement search functionality where users can search for movies based on genre.
- **Advanced Image Comparison:** Use more advanced models like ResNet-50 or fine-tune the existing model to improve accuracy.
- **Recommendation System:** Based on user preferences and movie ratings, suggest movies in addition to similarity-based results.
  
## Contributing

Feel free to fork the repository and submit pull requests. Contributions are welcome!
