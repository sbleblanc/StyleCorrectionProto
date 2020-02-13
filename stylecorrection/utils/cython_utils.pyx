import cython
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
def map_corpus(int[::1] corpus, int[::1] mapping):
    cdef Py_ssize_t length = corpus.shape[0]
    cdef Py_ssize_t i
    for i in prange(length, nogil=True):
        corpus[i] = mapping[corpus[i]]

@cython.boundscheck(False)
@cython.wraparound(False)
def count_words(int[::1] corpus, long[:,::1] positions, int[::1] counter):
    cdef Py_ssize_t length = positions.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    with nogil:
        for i in range(length):
            for j in range(positions[i][0], positions[i][1]):
                counter[corpus[j]] += 1