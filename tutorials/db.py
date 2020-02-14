import numpy as np
import skimage

if __name__ == '__main__':
    from trainer.lib.data_api import Session, ImStack, SemSegMask, reset_schema

    reset_schema()
    session = Session()

    # Binaries
    im_stack = ImStack.build_new(src_im=np.zeros((40, 28, 28, 1), dtype=np.uint8))
    mask1 = SemSegMask.build_new(gt_arr=np.zeros((28, 28, 3), dtype=np.bool))
    mask2 = SemSegMask.build_new(gt_arr=np.zeros((28, 28, 3), dtype=np.bool))
    im_stack.semseg_masks.extend([mask1, mask2])
    im_stack.set_class('digit', 'one')
    im_stack.set_class('nice', 'no')
    session.add(im_stack)

    # Binaries
    im_stack2 = ImStack.build_new(skimage.data.astronaut())
    mask1 = SemSegMask.build_new(np.zeros_like(skimage.data.astronaut(), dtype=np.bool))
    im_stack2.semseg_masks.extend([mask1, mask2])
    im_stack2.set_class('digit', 'two')
    session.add(im_stack2)

    session.commit()
