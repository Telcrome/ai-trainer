import os

import numpy as np

if __name__ == '__main__':
    if os.path.exists('./test.db'):
        print("Deleting old database")
        os.remove('./test.db')
    from trainer.lib.data_api import Session, ImageStack

    session = Session()
    # one_dim = NumpyBinary.from_ndarray(np.array([1, 2, 3]))
    # multi_dim = NumpyBinary.from_ndarray(np.random.rand(5, 5))
    # session.add(one_dim)
    # session.add(multi_dim)
    src_im = np.zeros((40, 28, 28, 1), dtype=np.uint8)

    im_stack = ImageStack.build_new(src_im=src_im)
    session.add(im_stack)
    session.commit()
