"""
This tutorial aims at demonstrating the impressive loading speed that trainer allows,
even on its (possibly) complex data structure!
"""

from sqlalchemy.orm import joinedload, subqueryload

import trainer.lib as lib
import trainer.lib.demo_data as demo_data

if __name__ == '__main__':
    # lib.reset_database()
    sess = lib.Session()
    ds = sess.query(lib.Dataset).filter(lib.Dataset.name == 'mnist').first()
    sd = demo_data.SourceData('D:\\')
    if ds is None:
        ds = demo_data.build_mnist(sd)

    from trainer.ml.torch_utils import bench_mark_dataset

    split_old = sess.query(lib.Split).options(subqueryload(lib.Split.sbjts)).first()
    split = sess.query(lib.Split).options(
        joinedload(lib.Split.sbjts).joinedload(lib.Subject.ims, innerjoin=True)
    ).filter(lib.Split.name == "train").first()
    print(split.name)

    mnist_res = bench_mark_dataset(sd.mnist_train, lambda t: (t[0].size, t[1]))
    trainer_res = bench_mark_dataset(split, lambda s: (s.ims[0].get_ndarray().shape, s.ims[0].get_class('digit')))
