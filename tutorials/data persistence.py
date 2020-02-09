import os
import shutil

import numpy as np
import skimage

import trainer.lib as lib

if __name__ == '__main__':
    data_path = r'C:\Users\rapha\Desktop\data'

    if os.path.exists(os.path.join(data_path, 'e1')):
        print("Removing old e1")
        shutil.rmtree(os.path.join(data_path, 'e1'))

    e1 = lib.Entity('e1', ['asdf', 'qwer'], data_path)
    e1.load_attr('asdf')['a'] = 5

    e2 = e1.create_child('e2', ['asdf', 'qwer'])
    e2.load_attr('asdf')['a'] = 5

    e1.to_disk()

    test_bin = skimage.data.astronaut()
    e2.add_bin('astronaut', test_bin, b_type=lib.BinaryType.NumpyArray.value)

    # e_load = lib.Entity.from_disk(os.path.join(data_path, 'e1'))
