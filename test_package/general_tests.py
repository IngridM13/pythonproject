from configs.settings import HDC_DIM
from encoding_methods.by_data_type.bool import BoolEncoding
from encoding_methods.by_data_type.numbers import IntegerEncoding, DecimalEncoding
from encoding_methods.by_data_type.type_datetime import DatetimeEncoding


def bool_test():
    enc = BoolEncoding(D=10000, flips_per_step=5, seed=42)

    # Codificar valores booleanos
    h_false = enc.encode(False)  # equivalente a enc.encode(0)
    h_true = enc.encode(True)  # equivalente a enc.encode(1)
    return h_true, h_false

    # print("Dimensión:", h_false.shape[0])
    # print("Ejemplo de componentes:", h_false[:10])
    # print("Valores posibles:", set(h_false))  # {-1, 1}
    #
    # # Similitudes
    # print("sim(False, False) =", enc.similarity(False, False))
    # print("sim(True, True)   =", enc.similarity(True, True))
    # print("sim(False, True)  =", enc.similarity(False, True))
    #
    # # También acepta 0/1
    # print("sim(0, 1)         =", enc.similarity(0, 1))
    #
    # # Accesores convenientes
    # hv0 = enc.false()  # hipervector canónico para False
    # hv1 = enc.true()  # hipervector canónico para True
    # print("sim(false(), true()) =", enc.similarity(hv0, hv1))

def decimal_test():
    dec_enc = DecimalEncoding(D=HDC_DIM, seed=123, x0=0.0, smoothness=10.0, omega_spread=4.0)
    a = dec_enc.encode(2.5)
    b = dec_enc.encode(2.6)
    c = dec_enc.encode(9.5)
    print("cos(2.5,2.6) =", dec_enc.similarity(2.5, 2.6))
    print("cos(2.5,9.5) =", dec_enc.similarity(2.5, 9.5))

def integer_test():
    # Integer -> Con Linear Mapping Encoding
    int_enc = IntegerEncoding(D=HDC_DIM, flips_per_step=5, n0=0, seed=42)
    h0 = int_enc.encode(0)
    h1 = int_enc.encode(1)
    h2 = int_enc.encode(2)
    print("cos(H0,H1) =", int_enc.similarity(0, 1))
    print("cos(H0,H2) =", int_enc.similarity(0, 2))
    print("cos(H1,H2) =", int_enc.similarity(1, 2))

def datetime_test():
    import datetime as dt
    from configs.settings import HDC_DIM, DEFAULT_SEED

    print("Ejemplo: DatetimeEncoding")
    enc = DatetimeEncoding(
        D=HDC_DIM,
        seed=DEFAULT_SEED,
        components=("year", "month", "day", "hour"),
    )

    # Fechas de prueba
    t_ref = dt.datetime(2024, 3, 15, 10, 0, 0)  # 15/03/2024 10:00
    t_next_month = dt.datetime(2024, 4, 15, 10, 0, 0)  # mes siguiente
    t_far = dt.datetime(2024, 9, 15, 10, 0, 0)  # varios meses después

    # Codificación
    h_ref = enc.encode(t_ref)
    h_next = enc.encode(t_next_month)
    h_far = enc.encode(t_far)

    # Similitudes (coseno)
    from encoding_methods.by_data_type.operaciones_vectoriales import cosine_similarity
    print("Dimensión HV:", h_ref.shape[0])
    print("sim(ref, ref)         =", cosine_similarity(h_ref, h_ref))
    print("sim(ref, mes+1)       =", cosine_similarity(h_ref, h_next))
    print("sim(ref, mes lejano)  =", cosine_similarity(h_ref, h_far))

    # También puedes codificar con diccionarios o tuplas
    # Diccionario explícito de componentes
    h_dict = enc.encode({"year": 2024, "month": 3, "day": 15, "hour": 10})
    print("sim(ref, dict mismo)  =", cosine_similarity(h_ref, h_dict))

    # Tupla (año, mes, día, hora)
    h_tuple = enc.encode((2024, 3, 15, 10))
    print("sim(ref, tupla misma) =", cosine_similarity(h_ref, h_tuple))

def string_test():
    # Ejemplos de codificación de strings: caracteres, palabras y textos (n-grams)
    from encoding_methods.by_data_type.strings import StringEncoding

    print("Ejemplo: StringEncoding (bipolar)")
    enc_bp = StringEncoding(D=HDC_DIM, seed=42, mode="bipolar", ngram_n=3, strategy="RI")

    # Caracteres
    hv_a = enc_bp.encode_char("a")
    hv_b = enc_bp.encode_char("b")
    print("sim_char(a,a) =", enc_bp.similarity(hv_a, hv_a))
    print("sim_char(a,b) =", enc_bp.similarity(hv_a, hv_b))

    # Palabras
    w1, w2, w3 = "casa", "caso", "perro"
    hv_w1 = enc_bp.encode_word(w1)
    hv_w2 = enc_bp.encode_word(w2)
    hv_w3 = enc_bp.encode_word(w3)
    print("sim_word(casa, caso)  =", enc_bp.similarity(hv_w1, hv_w2))
    print("sim_word(casa, perro) =", enc_bp.similarity(hv_w1, hv_w3))

    # Textos con n-grams (token-level)
    tokens1 = "el gato negro duerme".split()
    tokens2 = "el perro negro descansa".split()
    hv_t1 = enc_bp.encode_tokens_ngram(tokens1)  # usa n=3 por defecto
    hv_t2 = enc_bp.encode_tokens_ngram(tokens2)
    print("sim_text(tokens1, tokens2) =", enc_bp.similarity(hv_t1, hv_t2))

    print("Ejemplo: StringEncoding (binario)")
    enc_bin = StringEncoding(D=HDC_DIM, seed=123, mode="binary", ngram_n=2, strategy="RI")
    hv_w1_bin = enc_bin.encode_word(w1)
    hv_w2_bin = enc_bin.encode_word(w2)
    print("sim_word_bin(casa, caso) =", enc_bin.similarity(hv_w1_bin, hv_w2_bin))

#string_test()