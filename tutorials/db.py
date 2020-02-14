import os

import numpy as np

if __name__ == '__main__':
    if os.path.exists('./test.db'):
        print("Deleting old database")
        os.remove('./test.db')
    from trainer.lib.data_api import Session, ImageStack, SemSegMask

    session = Session()

    im_stack = ImageStack.build_new(src_im=np.zeros((40, 28, 28, 1), dtype=np.uint8))
    mask1 = SemSegMask.build_new(gt_arr=np.zeros((28, 28, 3), dtype=np.bool))
    mask2 = SemSegMask.build_new(gt_arr=np.zeros((28, 28, 3), dtype=np.bool))
    im_stack.semseg_masks.extend([mask1, mask2])
    session.add(im_stack)
    session.commit()
