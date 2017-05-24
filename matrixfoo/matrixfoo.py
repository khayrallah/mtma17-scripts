import numpy as np
import time

def get_toy_matrices():
    vocab_size_s = 4

    wordmap_s = {"wörter": 0, "worte": 1, "wegen": 2, "weil": 3}
    wordmap_t = {"words": 0, "because": 1, "as": 2, "foo": 3}

    similarities_raw = {
        "wörter": [("wörter", 1.0), ("worte", 0.9), ("wegen", 0.0), ("weil", 0.0)],
        "worte":  [("wörter", 0.4), ("worte", 1.0), ("wegen", 0.0), ("weil", 0.0)],
        "wegen":  [("wörter", 0.0), ("worte", 0.0), ("wegen", 1.0), ("weil", 0.6)],
        "weil":   [("wörter", 0.0), ("worte", 0.0), ("wegen", 0.6), ("weil", 1.0)],
    }

    # We could assume symmetry, but... eh.
    similarity_matrix = np.zeros((vocab_size_s, vocab_size_s))

    for (w1, l) in similarities_raw.items():
        for (w2, score) in l:
            similarity_matrix[wordmap_s[w1], wordmap_s[w2]] = score

    # first axis: V_S, second axis: V_T
    translation_matrix = np.zeros((vocab_size_s, vocab_size_s))
    
    # Populate initial matrix
    with open("/home/sjm/documents/MTMA2017/mtma17-scripts/matrixfoo/miniphrasetable") as f:
        for line in f.read().splitlines():
            [source, target, score, _] = line.split(" ||| ")
            i_s = wordmap_s[source]
            i_t = wordmap_t[target]
            translation_matrix[i_s, i_t] = score
    
    return (translation_matrix, similarity_matrix)

def get_random_matrices(size):
    similarity_matrix = np.random.rand(size, size)
    translation_matrix = np.random.rand(size, size)
    return (translation_matrix, similarity_matrix)

def crunch_matrices(translation_matrix, similarity_matrix):
    # Global normalize (useless?)
    #translation_matrix /= translation_matrix.sum()
    similarity_matrix /= similarity_matrix.sum(axis=1)[:, np.newaxis]

    for i in range(5):
        print(f"Iter {i}: ", end='')
        start = time.time()
        translation_matrix = np.dot(similarity_matrix, translation_matrix)
        #translation_matrix /= translation_matrix.sum()
        print(translation_matrix.sum(), end='')
        #print(translation_matrix)
        end = time.time()
        print(f" ({end - start} s)")


print("Generating matrices: ", end='')
start = time.time()
t, s = get_random_matrices(2000)
end = time.time()
print(f"{end - start} s")
crunch_matrices(t, s)