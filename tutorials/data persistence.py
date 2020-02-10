import os
import shutil

import trainer.lib as lib
import trainer.lib.demo_data as demo_data

if __name__ == '__main__':
    data_path = r'C:\Users\rapha\Desktop\data'
    storage_path = 'D:\\'
    sd = demo_data.SourceData(storage_path)

    e_name = 'd1'
    if os.path.exists(os.path.join(data_path, e_name)):
        print(f"Removing old {e_name}")
        shutil.rmtree(os.path.join(data_path, e_name))
        while os.path.exists(os.path.join(data_path, e_name)):
            pass

    d = lib.Dataset.build_new(e_name, data_path)

    s = demo_data.build_random_subject(sd)

    d.save_subject(s)

    # s = build_random_subject(d, sd)
    # d.save_subject(s)
    # d_load = lib.Dataset.from_disk(os.path.join(data_path, e_name))

    # s = demo_data.build_random_subject(d, sd)
