from typing import Tuple

import numpy as np
import torch.nn as nn
import torch.optim as optim

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
    im_stack = ml.image_stack_classification_preprocessor(s, 'digit')
    x, y = np.rollaxis(im_stack.get_src()[0], 2), np.array(int(im_stack.get_class_value('digit')))
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
    net.evaluate(test_set, acc_metric)


if __name__ == '__main__':
    data_path = r'C:\Users\rapha\Desktop\data'
    sd = demo_data.SourceData('D:\\')
    ds = demo_data.build_mnist(data_path, sd)

    class_name = 'digit'  # The structure that we now train for
    train_set = ml.TorchDataset(
        ds.get_working_directory(),
        f=preprocessor,
        split='train',
        mode=ml.ModelMode.Train
    )
    test_set = ml.TorchDataset(
        ds.get_working_directory(),
        f=preprocessor,
        split='test',
        mode=ml.ModelMode.Eval
    )

    BATCH_SIZE = 64
    EPOCHS = 1

    x = 0

    net = MnistNetwork()

    data_loader = train_set.get_torch_dataloader(batch_size=BATCH_SIZE, num_workers=4)

    acc_metric = ml.AccuracyMetric()
    train()
