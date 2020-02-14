import numpy as np
import skimage

if __name__ == '__main__':
    from trainer.lib.data_api import Session, ImStack, SemSegTpl, reset_database, MaskType, Subject

    reset_database()
    session = Session()

    # Semsegtpl
    ss_tpl = SemSegTpl.build_new(
        'basic',
        {
            "foreground": MaskType.Blob,
            "background": MaskType.Blob
        }
    )

    # Binaries
    im_stack = ImStack.build_new(src_im=np.zeros((40, 28, 28, 1), dtype=np.uint8))
    mask1 = im_stack.add_ss_mask(gt_arr=np.zeros((28, 28, 2), dtype=np.bool), sem_seg_tpl=ss_tpl)
    mask2 = im_stack.add_ss_mask(gt_arr=np.zeros((28, 28, 2), dtype=np.bool), sem_seg_tpl=ss_tpl)
    im_stack.semseg_masks.extend([mask1, mask2])
    im_stack.set_class('digit', 'one')
    im_stack.set_class('nice', 'no')
    session.add(im_stack)

    # Binaries
    im_stack2 = ImStack.build_new(skimage.data.astronaut())
    mask1 = im_stack2
    im_stack2.semseg_masks.append(mask1)
    im_stack2.set_class('digit', 'two')
    session.add(im_stack2)

    # Subject
    s = Subject.build_new('subject1')
    # s.ims.append(im_stack)

    session.commit()
