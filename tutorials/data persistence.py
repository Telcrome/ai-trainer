import os
import shutil

import trainer.lib as lib

if __name__ == '__main__':
    data_path = r'C:\Users\rapha\Desktop\data'
    e_name = 'd1'

    if os.path.exists(os.path.join(data_path, e_name)):
        print(f"Removing old {e_name}")
        shutil.rmtree(os.path.join(data_path, e_name))

    d = lib.Dataset(e_name, data_path)

    s = lib.Subject('s1')
    d.save_subject(s)
