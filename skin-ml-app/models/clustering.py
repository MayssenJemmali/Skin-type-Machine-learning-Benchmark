import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Skin_Type_dataset.csv')
CLUSTER_COLORS = ['#6c63ff', '#00d4aa', '#ff6b6b', '#ffd93d', '#c084fc', '#38bdf8']

# Unified mappings (Must match other modules)
LABEL_MAPS = {
    'Gender':         {'Female': 0, 'Male': 1},
    'Hydration_Level':{'High': 0, 'Low': 1, 'Medium': 2},
    'Oil_Level':      {'High': 0, 'Low': 1, 'Medium': 2},
    'Sensitivity':    {'High': 0, 'Low': 1, 'Medium': 2},
}
FEATURE_COLS = ['Age', 'Gender_enc', 'Hydration_Level_enc', 'Oil_Level_enc', 'Sensitivity_enc', 'Humidity', 'Temperature']

def load_data():
    df = pd.read_csv(DATA_PATH)
    for col, mapping in LABEL_MAPS.items():
        df[col + '_enc'] = df[col].map(mapping)
    X = df[FEATURE_COLS].values
    return X, df, FEATURE_COLS

def encode_input(inp):
    return [
        float(inp.get('Age', 30)),
        LABEL_MAPS['Gender'].get(inp.get('Gender', 'Female'), 0),
        LABEL_MAPS['Hydration_Level'].get(inp.get('Hydration_Level', 'Medium'), 2),
        LABEL_MAPS['Oil_Level'].get(inp.get('Oil_Level', 'Medium'), 2),
        LABEL_MAPS['Sensitivity'].get(inp.get('Sensitivity', 'Low'), 1),
        float(inp.get('Humidity', 50.0)),
        float(inp.get('Temperature', 25.0)),
    ]

def _cluster_profiles(df, labels):
    df_c = df.copy()
    df_c['cluster'] = labels
    profiles = []
    for lbl in sorted(set(labels)):
        if lbl == -1: continue
        sub = df_c[df_c['cluster'] == lbl]
        profiles.append({
            'cluster': int(lbl) + 1,
            'size': int(len(sub)),
            'dominant_skin_type': sub['Skin_Type'].mode()[0] if len(sub) > 0 else 'N/A',
            'avg_age': round(float(sub['Age'].mean()), 1),
            'avg_skin_score': round(float(sub['skin_score'].mean()), 1),
            'color': CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)],
        })
    return profiles

_MODEL_CACHE = {}

def predict_cluster(algorithm, encoded_features, params=None):
    cache_key = f"{algorithm}_{str(params)}"
    X, df, _ = load_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if cache_key in _MODEL_CACHE:
        model, labels = _MODEL_CACHE[cache_key]
    else:
        if algorithm == 'kmeans':
            k = int((params or {}).get('n_clusters', 4))
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
        else: # dbscan
            eps = float((params or {}).get('eps', 0.8))
            ms = int((params or {}).get('min_samples', 10))
            model = DBSCAN(eps=eps, min_samples=ms)
        
        labels = model.fit_predict(X_scaled)
        _MODEL_CACHE[cache_key] = (model, labels)
    
    x_in = scaler.transform([encoded_features])
    
    if algorithm == 'kmeans':
        cluster_idx = int(model.predict(x_in)[0])
    else: # dbscan logic
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1).fit(X_scaled)
        _, idx = nn.kneighbors(x_in)
        cluster_idx = labels[idx[0][0]]

    profiles = _cluster_profiles(df, labels)
    matched = next((p for p in profiles if p['cluster'] == cluster_idx + 1), None)
    
    return {
        'cluster_name': f'Group #{cluster_idx + 1}' if cluster_idx != -1 else 'Unique (Outlier)',
        'profile': matched,
        'algorithm': algorithm.upper()
    }

def run_clustering(algorithm, params=None):
    X, df, _ = load_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=int((params or {}).get('n_clusters', 4)), random_state=42, n_init=10)
    else:
        model = DBSCAN(eps=float((params or {}).get('eps', 0.8)), min_samples=int((params or {}).get('min_samples', 10)))
    
    labels = model.fit_predict(X_scaled)
    profiles = _cluster_profiles(df, labels)
    
    return {
        'n_clusters': int(len(set(labels)) - (1 if -1 in labels else 0)),
        'profiles': profiles
    }
