
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.sparse import csr_matrix

def item_based_knn_train(df_train, k=5):
    df_train["relevance"] = 5 * df_train["booking_bool"] + df_train["click_bool"]
    item_encoder = LabelEncoder()
    encoded_items = item_encoder.fit_transform(df_train["prop_id"])
    encoded_users = LabelEncoder().fit_transform(df_train["srch_id"])

    interaction_matrix = csr_matrix(
        (df_train["relevance"], (encoded_items, encoded_users)),
        shape=(len(np.unique(encoded_items)), len(np.unique(encoded_users)))
    )

    model = NearestNeighbors(n_neighbors=k+1, metric="cosine", algorithm="brute")
    model.fit(interaction_matrix)

    item_id_to_index = {pid: item_encoder.transform([pid])[0] for pid in df_train["prop_id"].unique()}
    index_to_item_id = {v: k for k, v in item_id_to_index.items()}

    return model, item_id_to_index, index_to_item_id

def score_test_using_item_knn(df_test, model, item_id_to_index, index_to_item_id, df_train):
    df_test["relevance"] = 5 * df_test.get("booking_bool", 0) + df_test.get("click_bool", 0)
    df_test["item_knn_score"] = 0.0
    popularity = df_train.groupby("prop_id")["relevance"].mean().to_dict()

    for i, row in df_test.iterrows():
        prop_id = row["prop_id"]
        if prop_id in item_id_to_index:
            idx = item_id_to_index[prop_id]
            distances, indices = model.kneighbors([model._fit_X[idx].toarray().flatten()])
            similar_props = [index_to_item_id[ind] for ind in indices.flatten() if ind != idx]
            score = df_train[df_train["prop_id"].isin(similar_props)]["relevance"].mean()
            if pd.isna(score):
                score = popularity.get(prop_id, 0)
        else:
            score = popularity.get(prop_id, 0)
        df_test.at[i, "item_knn_score"] = score

    df_test["item_knn_score"] = MinMaxScaler().fit_transform(df_test[["item_knn_score"]])
    return df_test

def dcg(relevances, k):
    relevances = np.array(relevances)[:k]
    return np.sum((2 ** relevances - 1) / np.log2(np.arange(2, len(relevances) + 2)))

def ndcg_at_k(relevances, k):
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal, k)
    return dcg(relevances, k) / ideal_dcg if ideal_dcg > 0 else 0.0

def mean_ndcg_at_k(df, score_column="item_knn_score", k=5):
    df = df.sort_values(["srch_id", score_column], ascending=[True, False])
    ndcgs = []
    for _, group in df.groupby("srch_id"):
        relevances = group["relevance"].tolist()
        ndcgs.append(ndcg_at_k(relevances, k))
    return np.mean(ndcgs)

if __name__ == "__main__":
    df = pd.read_parquet("data/train_processed.parquet")
    df_sample = df.sample(frac=0.1, random_state=42)

    split = int(0.8 * len(df_sample))
    df_train = df_sample[:split].copy()
    df_val = df_sample[split:].copy()

    model, item_id_to_index, index_to_item_id = item_based_knn_train(df_train, k=5)
    df_val = score_test_using_item_knn(df_val, model, item_id_to_index, index_to_item_id, df_train)

    ndcg = mean_ndcg_at_k(df_val, score_column="item_knn_score", k=5)
    print(f"Validation NDCG@5: {ndcg:.4f}")

    df_val.sort_values(by=["srch_id", "item_knn_score"], ascending=[True, False], inplace=True)
    df_val[["srch_id", "prop_id"]].to_csv("item_knn_prediction_table.csv", index=False)
