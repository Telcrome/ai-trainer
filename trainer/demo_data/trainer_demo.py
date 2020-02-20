"""
Defines a dataset that showcases the learning capabilities of trainer.
"""
from typing import Tuple

import sqlalchemy as sa

import trainer.lib as lib
import trainer.demo_data as dd


class DemoDataset(dd.DemoDataset):

    def __init__(self):
        super().__init__('', ds_name='Demo')

    def add_simple_split(self, split_name='simple'):
        pass

    def build_dataset(self, sess=None) -> Tuple[lib.Dataset, sa.orm.session.Session]:
        d, sess = super().build_dataset(sess)

        ss_tpl = lib.SemSegTpl.build_new(
            'demo_colors',
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

        return d, sess


if __name__ == '__main__':
    lib.reset_database()
    ds, s = DemoDataset().build_dataset()
