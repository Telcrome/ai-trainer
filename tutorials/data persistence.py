import os
import shutil

import trainer.lib as lib

if __name__ == '__main__':
    data_path = r'C:\Users\rapha\Desktop\data'
    e_name = 'd1'

    if os.path.exists(os.path.join(data_path, e_name)):
        print(f"Removing old {e_name}")
        shutil.rmtree(os.path.join(data_path, e_name))

    d = lib.Dataset.build_new(e_name, data_path)
    # e1.add_attr('asdf', {'a': 5})
    #
    # e2 = e1.create_child('e2')
    # e2.add_attr('qwer')
    # e2.load_attr('qwer')['a'] = 5
    #
    # test_bin = skimage.data.astronaut()
    # e2.add_bin('astronaut', test_bin, b_type=lib.BinaryType.NumpyArray.value)
    # e1.add_bin('e1stronaut', test_bin, b_type=lib.BinaryType.Unknown.value)
    # e1.to_disk()
    #
    # e_load = lib.Entity.from_disk(os.path.join(data_path, 'e1'))
    # c2 = e_load.get_child('e2')
