# borrador - sería para HDVs con valores reales
from scipy.stats import cosine


def find_similar_rows(hdv_query, df, threshold=0.1):
    """Find rows with HDVs similar to the query."""
    df["similarity"] = df["HDV"].apply(lambda x: 1 - cosine(hdv_query, x))
    return df[df["similarity"] >= threshold].sort_values(by="similarity", ascending=False)

#query_hdv = df.iloc[0]["HDV"]  # Use the first row as an example query
#matches = find_similar_rows(query_hdv, df, threshold=0.85)
#print(matches[["name", "lastname", "similarity"]])
