from typing import Tuple

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

import trainer.lib as lib
import trainer.lib.demo_data as demo_data
import trainer.ml as ml
from trainer.ml.multi_network import SmallClassNet


class MnistNetwork(ml.TrainerModel):
    def __init__(self):
        model = SmallClassNet()
        opti = optim.Adam(model.parameters(), lr=5e-3)
        crit = nn.CrossEntropyLoss()
        visboard = ml.VisBoard(run_name=lib.create_identifier('MnistNetwork'))
        super().__init__(model_name='multi_model',
                         model=model,
                         opti=opti,
                         crit=crit,
                         visboard=visboard)


def preprocessor(s: lib.Subject, m: ml.ModelMode) -> Tuple[np.ndarray, np.ndarray]:
    # noinspection PyUnresolvedReferences
    im_stack = s.ims[0]
    x, y = np.rollaxis(im_stack.get_ndarray()[0], 2), np.array(int(im_stack.get_class('digit')))
    return x.astype(np.float32) / 127.5 - 1, y


def train():
    for epoch in range(EPOCHS):
        net.run_epoch(
            data_loader,
            epoch=epoch,
            batch_size=BATCH_SIZE,
            steps=-1
        )


def test():
    acc_metric = ml.AccuracyMetric()
    test_loader = test_set.get_torch_dataloader(batch_size=64, num_workers=2)
    net.evaluate(test_loader, acc_metric)
    print(acc_metric.get_result())


if __name__ == '__main__':
    # lib.reset_database()
    sess = lib.Session()
    ds = sess.query(lib.Dataset).filter(lib.Dataset.name == 'mnist').first()
    sd = demo_data.SourceData('D:\\')
    if ds is None:
        ds = demo_data.build_mnist(sd)

    class_name = 'digit'  # The structure that we now train for
    train_set = ml.TorchDataset(
        'mnist',
        'train',
        f=preprocessor,
        mode=ml.ModelMode.Train,
        in_memory=True
    )
    test_set = ml.TorchDataset(
        'mnist',
        'test',
        f=preprocessor,
        mode=ml.ModelMode.Eval
    )

    BATCH_SIZE = 32
    EPOCHS = 5

    x = 0

    net = MnistNetwork()

    data_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE)
    # data_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=BATCH_SIZE, shuffle=True)

    train()
    test()
