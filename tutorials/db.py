import numpy as np
import skimage

import trainer.lib as lib

if __name__ == '__main__':
    lib.reset_database()
    session = lib.Session()

    ss_tpl = lib.SemSegTpl.build_new(
        'us_bone',
        {
            "femur": lib.MaskType.Line,
            "tibia": lib.MaskType.Line
        }
    )

    cls_def = lib.ClassDefinition.build_new(
        'cruciate_ligament_state',
        cls_type=lib.ClassType.Ordinal,
        values=[
            'healthy',
            'partial_rupture',
            'rupture'
        ]
    )

    f_path, _ = lib.standalone_foldergrab(folder_not_file=True, title='Select the parent folder')
    d = lib.Dataset.build_new('US_BONE')
    d.add_split('default')
    lib.add_image_folder(d.splits[0], folder_path=f_path)

    # # Binaries
    # im_stack = lib.ImStack.build_new(src_im=np.zeros((40, 28, 28, 1), dtype=np.uint8))
    # mask1 = im_stack.add_ss_mask(gt_arr=np.zeros((28, 28, 2), dtype=np.bool), sem_seg_tpl=ss_tpl)
    # mask2 = im_stack.add_ss_mask(gt_arr=np.zeros((28, 28, 2), dtype=np.bool), sem_seg_tpl=ss_tpl)
    # im_stack.semseg_masks.extend([mask1, mask2])
    # im_stack.set_class('digit', 'one')
    # im_stack.set_class('nice', 'no')
    # session.add(im_stack)
    #
    # # Binaries
    # im_stack2 = ImStack.build_new(skimage.data.astronaut())
    # mask1 = im_stack2
    # im_stack2.semseg_masks.append(mask1)
    # im_stack2.set_class('digit', 'two')
    # session.add(im_stack2)
    #
    # # Subject
    # s = Subject.build_new('subject1')
    # s.ims.append(im_stack)

    session.commit()
