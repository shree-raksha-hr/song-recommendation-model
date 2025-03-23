from flask import Flask, render_template, request,  url_for, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

# Load the dataset
df=pd.read_csv('cleaned_dataset.csv')

# Select features and normalize
features = ['Danceability', 'Energy',
       'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness',
       'Liveness', 'Valence', 'Tempo']

# Apply PCA for Dimentionality reduction
pca = PCA(n_components=3)
X_pca = pca.fit_transform(df[features])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=40, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_pca)

# Function to get recommendations
def get_recommendations(song_name):
    if song_name not in df['Track'].values:
        return []

    # Find the cluster of the given song
    song_cluster_num = df[df['Track'] == song_name]['cluster'].values[0]
    print(song_cluster_num)

    #Cluster of the selected song    
    recommendations = df[df['cluster'] == song_cluster_num].copy()

        #Get the index of the song
    song_idx = recommendations[recommendations['Track'] == song_name].index[0]

    # Apply NearestNeighbors on the recommended cluster
    knn = NearestNeighbors(n_neighbors=11, metric='cosine')
    knn.fit(recommendations[features])

    #Calculate distances and indices of the nearest neighbors
    distances, indices = knn.kneighbors([df.iloc[song_idx][features].values], n_neighbors=11)

    #Array of song name and track of the 10 nearest neighbors
    recs=[]
    for i in range(1, len(indices[0])):
        a={'Track':recommendations.iloc[indices[0][i]]['Track'], 'Artist':recommendations.iloc[indices[0][i]]['Artist']}
        recs.append(a)

    return recs

#Global variable to store the selected song
song=""

#----------------Routes------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    global song
    search_song=""
    recommendations = []

    # Get all unique song names for the dropdown
    song_list = df["Track"].unique().tolist()

    if request.method == "POST":
        search_song = request.form["song_name"]
        recommendations = get_recommendations(search_song)
        song=search_song
        print(song)

    return render_template("index.html", recommendations=recommendations, search_song=search_song, song_list=song_list)


def search_song():
    search_song=""
    if request.method == "POST":
        search_song = request.form["song_name"]
        recommendations = get_recommendations(search_song)




@app.route('/histograms')
def histograms():
    histograms_json = {}
    for feature in features:
        fig = px.histogram(df, x=feature, nbins=50, title=f"Distribution of {feature}")
        histograms_json[feature] = pio.to_json(fig)
    return jsonify(histograms_json)


@app.route('/heatmap')
def heatmap():
    corr_matrix = df[features].corr()
    fig = px.imshow(
        corr_matrix, 
        labels=dict(x="Features", y="Features", color="Correlation"),
        x=features, 
        y=features,
        color_continuous_scale="Viridis",
        title="Feature Correlation Heatmap"
    )
    return jsonify(pio.to_json(fig))


@app.route('/pca_kmeans')
def pca_plot_3d():
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(df[features])

    df_pca = pd.DataFrame(X_pca_3d, columns=['PCA1', 'PCA2', 'PCA3'])
    df_pca['Cluster'] = df['cluster'].astype(str)
    fig = px.scatter_3d(df_pca, x='PCA1', y='PCA2', z='PCA3', color='Cluster',
                         title='3D PCA Visualization of Clusters',
                         labels={'Cluster': 'Cluster Number'},
                         opacity=0.8)

    return jsonify(pio.to_json(fig))


@app.route('/tsne_plot')
def tsne_plot():
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(df[features])

    df_tsne = pd.DataFrame(X_tsne, columns=['t-SNE1', 't-SNE2'])
    df_tsne['Cluster'] = df['cluster'].astype(str)
    fig = px.scatter(df_tsne, x='t-SNE1', y='t-SNE2', color='Cluster',
                     title='t-SNE Visualization of Clusters',
                     labels={'Cluster': 'Cluster Number'},
                     opacity=0.7)
    
    return jsonify(pio.to_json(fig))



@app.route('/elbow_plot')
def elbow_plot():
    inertia_values = []
    k_values = list(range(10,151,10))
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df[features])
        inertia_values.append(kmeans.inertia_)
    fig = px.line(x=k_values, y=inertia_values, markers=True,
                  title="Elbow Method for Optimal K",
                  labels={"x": "Number of Clusters (K)", "y": "Inertia"})

    return jsonify(pio.to_json(fig))


@app.route('/nearest_songs_plot')
def nearest_songs_plot():
    song_name=song
    print(f"Selected song: {song_name}")

    kmeans = KMeans(n_clusters=40, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_pca)
    
    if song_name not in df['Track'].values:
        return []

    song_cluster_num = df[df['Track'] == song_name]['cluster'].values[0]
    print(song_cluster_num)
        
    recommendations = df[df['cluster'] == song_cluster_num].copy()

    knn = NearestNeighbors(n_neighbors=11, metric='cosine')
    knn.fit(recommendations[features])

    song_idx = recommendations[recommendations['Track'] == song_name].index[0]
    distances, indices = knn.kneighbors([df.iloc[song_idx][features].values], n_neighbors=11)

    x=[]
    for i in range(1, len(indices[0])):
        a={'Track':recommendations.iloc[indices[0][i]]['Track'], 'Distance':distances[0][i]}
        x.append(a)

    df1=pd.DataFrame(x)

    fig = px.bar(df1, x="Track", y="Distance", text="Distance",
             labels={"Track": "Recommended Songs", "Distance": "Distance"},
             title="Nearest Songs Based on Distance")

    print(jsonify(pio.to_json(fig)))
    
    return jsonify(pio.to_json(fig))


label=kmeans.labels_
silhouette = silhouette_score(X_pca,label)
ch_score = calinski_harabasz_score(X_pca,label)
db_score = davies_bouldin_score(X_pca,label)


@app.route('/model_info')
def model_info():
    return render_template("model_info.html",
                           silhouette_score=silhouette, ch_score=ch_score, db_score=db_score)

if __name__ == "__main__":
    app.run(debug=True, port=5050)
