import os

import numpy as np

if __name__ == '__main__':
    if os.path.exists('./test.db'):
        os.remove('./test.db')
    from trainer.lib.data_api import NumpyBinary, Session

    session = Session()
    one_dim = NumpyBinary.from_ndarray(np.array([1, 2, 3]))
    multi_dim = NumpyBinary.from_ndarray(np.random.rand(5, 5))
    session.add(one_dim)
    session.add(multi_dim)
    session.commit()
