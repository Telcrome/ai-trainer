from timeit import default_timer as timer

import numpy as np
from numba import njit, prange, jit, cuda
import matplotlib.pyplot as plt
import seaborn as sns


def cuda_comparison(res_array: np.ndarray, loops: int) -> None:
    for x in prange(res_array.shape[0]):
        for y in range(res_array.shape[1]):
            for i in range(loops):
                res_array[x, y] += np.cos(x) ** 2 + np.sin(x) ** 2


numba_f = njit(cuda_comparison)
numba_parallel = njit(cuda_comparison, parallel=True)


@cuda.jit()
def cuda_test(res_array: np.ndarray) -> None:
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < res_array.size:  # Check array boundaries
        res_array[pos] += 1


if __name__ == '__main__':
    LOOPS_N = 500
    DIM = 100

    res = np.random.random(size=(DIM, DIM))
    start = timer()
    cuda_comparison(res, LOOPS_N)
    end = timer()
    numpy_seconds = end - start

    res_numba = np.random.random(size=(DIM, DIM))
    start = timer()
    numba_f(res_numba, LOOPS_N)
    end = timer()
    numba_seconds = end - start

    res_cuda = np.random.random(size=(DIM, DIM))
    start = timer()
    numba_parallel(res_cuda, LOOPS_N)
    end = timer()
    cuda_seconds = end - start

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    sns.heatmap(res, ax=ax1)
    ax1.set_title(f"Pure: {numpy_seconds}")
    sns.heatmap(res_numba, ax=ax2)
    ax2.set_title(f"Numba: {numba_seconds}")
    sns.heatmap(res_cuda, ax=ax3)
    ax3.set_title(f"Parallel: {cuda_seconds}")
    plt.show()
