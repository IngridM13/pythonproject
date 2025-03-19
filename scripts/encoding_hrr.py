from scipy.signal import fftconvolve
# se puede hacer unbinding y obtener features del HDV

def circular_convolution(v1, v2):
    return fftconvolve(v1, v2, mode="same")  # HRR-style binding


def encode_row_hrr(row):
    """Encodes a single row using HRR (Holographic Reduced Representations)."""
    hdv = np.zeros(DIMENSIONS)  # Initialize HDV

    for col, value in row.items():
        if pd.notna(value):
            value_hdv = get_feature_hdv(str(value))
            hdv = circular_convolution(hdv, value_hdv)  # Bind with HRR

    return hdv / np.linalg.norm(hdv)  # Normalize


df["HDV"] = df.apply(encode_row_hrr, axis=1)
