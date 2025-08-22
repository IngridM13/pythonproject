from scipy.spatial.distance import hamming
# para uso en HDVs binarios
def find_binary_matches(query_hdv, df, threshold=0.9):
    """Find rows with HDVs that have low Hamming distance to the query."""
    df["hamming_distance"] = df["HDV"].apply(lambda x: 1 - hamming(query_hdv, x))
    return df[df["hamming_distance"] >= threshold].sort_values(by="hamming_distance", ascending=False)

# query_hdv = df.iloc[0]["HDV"]
# matches = find_binary_matches(query_hdv, df, threshold=0.9)
# print(matches[["name", "lastname", "hamming_distance"]])
