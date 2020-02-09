import os
import shutil

import trainer.lib as lib

if __name__ == '__main__':
    data_path = r'C:\Users\rapha\Desktop\data'

    if os.path.exists(os.path.join(data_path, 'e1')):
        print("Removing old e1")
        shutil.rmtree(os.path.join(data_path, 'e1'))

    e = lib.Entity('e1', data_path)
    e.to_disk()
