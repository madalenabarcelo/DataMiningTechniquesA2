import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

class ClusteredItemBasedCF:
    def __init__(self, k=5, n_clusters=50):
        self.k = k
        self.n_clusters = n_clusters
        self.clusters = None
        self.user_item_matrix = None
        self.item_similarity_per_cluster = {}
        self.item_to_cluster = {}

    def cluster_items(self, df, features, item_col='prop_id'):
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(df[features])
        
        # Cluster items
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X)
        self.clusters = df[[item_col, 'cluster']].drop_duplicates()
        self.item_to_cluster = dict(zip(self.clusters[item_col], self.clusters['cluster']))
    
        return self.clusters

    def format_data(self, train_df: pd.DataFrame):
        self.user_item_matrix = train_df.pivot_table(
            index="srch_id", columns="prop_id", values="click_bool", fill_value=0
        )

    def compute_similarity_per_cluster(self):
        from sklearn.metrics.pairwise import cosine_similarity

        cluster_groups = defaultdict(list)
        for item_id, cluster_id in self.item_to_cluster.items():
            if item_id in self.user_item_matrix.columns:
                cluster_groups[cluster_id].append(item_id)  # Only include seen items

        for cluster_id, items in cluster_groups.items():
            filtered_matrix = self.user_item_matrix[items].T  # Items as rows
            sim_matrix = pd.DataFrame(
                cosine_similarity(filtered_matrix),
                index=items,
                columns=items
            )
            self.item_similarity_per_cluster[cluster_id] = sim_matrix


    def format_and_train(self, train_df: pd.DataFrame, item_features: pd.DataFrame, feature_cols=None):
        if feature_cols is None:
            raise ValueError("You must provide a list of feature column names.")
        
        self.cluster_items(item_features, features=feature_cols)
        self.format_data(train_df)
        self.compute_similarity_per_cluster()

    def predict(self, user_id, item_id):
        cluster_id = self.item_to_cluster.get(item_id)
        if cluster_id is None or cluster_id not in self.item_similarity_per_cluster:
            return 0

        similarity_matrix = self.item_similarity_per_cluster[cluster_id]
        if item_id not in similarity_matrix:
            return 0

        user_ratings = self.user_item_matrix.loc[user_id]
        similar_items = similarity_matrix[item_id].drop(item_id, errors='ignore')
        interacted_items = user_ratings[user_ratings > 0]
        common_items = similar_items[interacted_items.index.intersection(similar_items.index)]

        top_k = common_items.sort_values(ascending=False).head(self.k)
        numerator = sum(user_ratings[i] * top_k[i] for i in top_k.index)
        denominator = sum(top_k)
        return numerator / denominator if denominator != 0 else 0

    def get_all_predictions(self, user_id):
        user_ratings = self.user_item_matrix.loc[user_id]
        predictions = {}
        for item_id in self.user_item_matrix.columns:
            if user_ratings[item_id] == 0:
                predictions[item_id] = self.predict(user_id, item_id)
        return sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    def predict_for_test_users(self, test_df):
        all_predictions = []
        test_user_ids = test_df['srch_id'].unique()

        for user_id in test_user_ids:
            if user_id in self.user_item_matrix.index:
                preds = self.get_all_predictions(user_id)
                for prop_id, score in preds:
                    all_predictions.append({
                        "srch_id": user_id,
                        "prop_id": prop_id,
                        "score": score
                    })
        return pd.DataFrame(all_predictions)
