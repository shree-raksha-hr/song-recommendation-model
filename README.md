# 🎵 Music Recommendation System

A machine learning-based recommendation engine that suggests similar songs based on audio features and popularity metrics. It uses **K-Means Clustering** to group similar tracks and **K-Nearest Neighbors (KNN)** to recommend songs that closely resemble a selected input.

---

## 📌 Features

- 🔍 **Content-Based Filtering** using audio and metadata attributes  
- 🎯 **K-Means Clustering** for grouping similar songs  
- 🤖 **K-Nearest Neighbors (KNN)** for finding song similarities  
- 📊 **Clustering Evaluation** with Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score  
- 📁 Handles large datasets efficiently  
- ✅ No need for user history or explicit feedback  

---

## 🧠 Algorithms Used

- **K-Means Clustering** – Unsupervised algorithm to group songs with similar characteristics.  
- **K-Nearest Neighbors (KNN)** – Supervised similarity-based algorithm to recommend songs.  
- **Feature Scaling** – Standardization of features to normalize input data.  

---

## 📑 Dataset

- Total records: **20,594 songs**
- Key features used:
  - `danceability`, `energy`, `loudness`, `tempo`, `speechiness`
  - `acousticness`, `instrumentalness`, `liveness`, `valence`
  - Popularity metrics: `views`, `likes`, `comments`

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/music-recommendation-system.git
cd music-recommendation-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Dataset
Place your dataset as `music_dataset.csv` in the root directory.

### 4. Run the System
```bash
python main.py
```

---

## 📈 Evaluation Metrics

- **Silhouette Score** – Measures how well samples are clustered.
- **Davies-Bouldin Index** – Lower values indicate better clustering.
- **Calinski-Harabasz Score** – Higher values indicate better cluster separation.

---

## 📬 Example Usage

```python
recommend_songs(song_index=100, n_recommendations=5)
```

Outputs the top 10 most similar songs based on audio features along with helpful visualizations about the dataset

---

## 🛠 Built With

- Python 3  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  

---

## 📌 Future Improvements

- 🎧 Add genre or mood filtering  
- 🌐 Build a web interface using Flask or Streamlit  
- 🔊 Integrate with APIs (e.g., Spotify Web API for real-time data)  
- 📱 Make it interactive using a GUI or mobile app
