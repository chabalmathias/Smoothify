cimport cython
import numpy as np
cimport numpy as cnp


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cnp.ndarray[cnp.double_t, ndim=2] _chaikin(double[:, ::1] points,
                                                         const int num_iterations,
                                                         bint is_closed):

    cdef Py_ssize_t it, i, k
    cdef Py_ssize_t n_in = points.shape[0] # number of points
    cdef Py_ssize_t dim = points.shape[1] # number of coordinates per point
    cdef Py_ssize_t n_out # number of output points after smoothing
    cdef double p0, p1

    cdef cnp.ndarray[cnp.double_t, ndim=2] result
    cdef double[:, ::1] current = points
    cdef double[:, ::1] nxt # next points

    # Memorize the endpoints.
    cdef double[::1] endpoint_start
    cdef double[::1] endpoint_end

    if not is_closed:
        # Store endpoints for open linestrings
        endpoint_start = np.array(points[0], dtype=np.float64)
        endpoint_end = np.array(points[n_in - 1], dtype=np.float64)

    for it in range(num_iterations):
        n_in = current.shape[0]

        if is_closed:
            n_out = n_in * 2
        else:
            # keeping the 2 extremities + 2 points per segment
            n_out = (n_in - 1) * 2 + 2

        result = np.empty((n_out, dim), dtype=np.float64, order='C')
        nxt = result

        if is_closed:
            for i in range(n_in):
                # segment i -> (i+1) % n_in
                for k in range(dim):
                    p0 = current[i, k]
                    p1 = current[(i + 1) % n_in, k]
                    nxt[2 * i, k] = 0.75 * p0 + 0.25 * p1  # point q
                    nxt[2 * i + 1, k] = 0.25 * p0 + 0.75 * p1  # point r

        else:
            for k in range(dim):
                nxt[0, k] = endpoint_start[k]
            for i in range(n_in - 1):
                for k in range(dim):
                    p0 = current[i, k]
                    p1 = current[i + 1, k]
                    nxt[2 * i + 1, k] = 0.75 * p0 + 0.25 * p1
                    nxt[2 * i + 2, k] = 0.25 * p0 + 0.75 * p1

            # endpoints conserved
            for k in range(dim):
                nxt[n_out - 1, k] = endpoint_end[k]

        current = nxt

    return np.asarray(current)
