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
    session.add(ss_tpl)

    cls_def = lib.ClassDefinition.build_new(
        'cruciate_ligament_state',
        cls_type=lib.ClassType.Ordinal,
        values=[
            'healthy',
            'partial_rupture',
            'rupture'
        ]
    )
    session.add(cls_def)

    d = lib.Dataset.build_new('US_BONE')

    # d.add_split('default')
    # f_path, _ = lib.standalone_foldergrab(folder_not_file=True, title='Select the parent folder')
    # lib.add_image_folder(d.splits[0], folder_path=f_path, sess=session)

    # d.add_split('imported')
    # f_path, _ = lib.standalone_foldergrab(folder_not_file=True, title='Select the import folder')
    # lib.add_import_folder(d.get_split_by_name('imported'), folder_path=f_path, semsegtpl=ss_tpl)

    session.add(d)

    session.commit()
