from functools import reduce
import json
import os
from typing import List

import numpy as np
from tqdm import tqdm

import trainer.demo_data as dd
import trainer.lib as lib


def array_from_json(list_repr: List[List], depth=10) -> np.ndarray:
    width, height = len(list_repr), len(list_repr[0])
    res = np.zeros((width, height, depth), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            depth_val = list_repr[x][y]
            res[x, y, depth_val] = 1
    return res


class ArcDataset(dd.DemoDataset):

    def __init__(self, data_path: str):
        super().__init__(data_path, 'arc')
        self.arc_path = os.path.join(self.kaggle_storage, 'abstraction-and-reasoning-challenge')
        if not os.path.exists(self.arc_path):
            raise FileNotFoundError("The files required for building the arc dataset could not be found")

    def create_arc_split(self, d: lib.Dataset, ss_tpl: lib.SemSegTpl, split_name='training'):
        d.add_split(split_name=split_name)
        p = os.path.join(self.arc_path, split_name)
        for file_path in tqdm(os.listdir(p), desc=split_name):
            f_name = os.path.split(file_path)[-1]
            s_name = os.path.splitext(f_name)[0]
            s = lib.Subject.build_new(s_name)
            with open(os.path.join(p, f_name), 'r') as f:
                json_content = json.load(f)
            for key in json_content:
                for maze in json_content[key]:
                    extra_info = {'purpose': key}
                    # Input Image
                    input_json = maze['input']
                    input_im = array_from_json(input_json, depth=10)
                    im_stack = lib.ImStack.build_new(src_im=input_im, extra_info=extra_info)

                    # If the solution is given, add:
                    if 'output' in maze:
                        output_json = maze['output']
                        gt_arr = array_from_json(output_json, depth=10).astype(np.bool)
                        im_stack.add_ss_mask(
                            gt_arr,
                            sem_seg_tpl=ss_tpl,
                            ignore_shape_mismatch=True)

                        # Add metadata
                        im_stack.extra_info['sizeeq'] = (input_im.shape == gt_arr.shape)

                    s.ims.append(im_stack)
            s.extra_info['all_have_target'] = reduce(lambda x, y: x and y,
                                                     ['sizeeq' in im.extra_info for im in s.ims])
            if s.extra_info['all_have_target']:
                s.extra_info['sizeeq'] = reduce(lambda x, y: x and y,
                                                [im.extra_info['sizeeq'] for im in s.ims])
            d.get_split_by_name(split_name).sbjts.append(s)

    def build_dataset(self, sess=None) -> lib.Dataset:
        d, sess = super().build_dataset(sess)

        # Dataset does not exist yet, build it!
        ss_tpl = lib.SemSegTpl.build_new(
            'arc_colors',
            {
                "Zero": lib.MaskType.Blob,
                "One": lib.MaskType.Blob,
                "Two": lib.MaskType.Blob,
                "Three": lib.MaskType.Blob,
                "Four": lib.MaskType.Blob,
                "Five": lib.MaskType.Blob,
                "Six": lib.MaskType.Blob,
                "Seven": lib.MaskType.Blob,
                "Eight": lib.MaskType.Blob,
                "Nine": lib.MaskType.Blob,
            }
        )
        sess.add(ss_tpl)

        self.create_arc_split(d, ss_tpl, 'training')
        self.create_arc_split(d, ss_tpl, 'test')
        self.create_arc_split(d, ss_tpl, 'evaluation')
        sess.add(d)
        sess.commit()
        return d


if __name__ == '__main__':
    lib.reset_database()
    arc = ArcDataset('D:\\')
    ds = arc.build_dataset()
