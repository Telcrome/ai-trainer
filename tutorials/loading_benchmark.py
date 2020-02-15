"""
This tutorial aims at demonstrating the impressive loading speed that trainer allows,
even on its (possibly) complex data structure!
"""

from sqlalchemy.orm import joinedload, subqueryload
from tqdm import tqdm

import trainer.lib as lib
import trainer.lib.demo_data as demo_data
import trainer.ml as ml
from trainer.ml.torch_utils import bench_mark_dataset


def benchmark_mnist():
    ds = sess.query(lib.Dataset).filter(lib.Dataset.name == 'mnist').first()
    if ds is None:
        ds = sd.build_mnist(sd)

    split_old = sess.query(lib.Split).options(subqueryload(lib.Split.sbjts)).first()
    split = sess.query(lib.Split).options(
        joinedload(lib.Split.sbjts).joinedload(lib.Subject.ims, innerjoin=True)
    ).filter(lib.Split.name == "train").first()
    print(split.name)

    mnist_res = bench_mark_dataset(sd.mnist_train, lambda t: (t[0].size, t[1]))
    trainer_res = bench_mark_dataset(split, lambda s: (s.ims[0].get_ndarray().shape, s.ims[0].get_class('digit')))


if __name__ == '__main__':
    # lib.reset_database()
    sess = lib.Session()
    sd = demo_data.SourceData('D:\\')

    ds = sd.build_arc(sess)

    split = ds.get_split_by_name('training')
    aux = []
    for s in tqdm(split):
        aux.append(s.ims[0].get_ndarray())

    aux = []
    for s, gt in tqdm(ml.InMemoryDataset('arc', 'training', lambda x, y: (x, y), mode=ml.ModelMode.Train)):
        aux.append(s.ims[0].get_ndarray())
