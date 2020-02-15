import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import trainer.lib as lib
import trainer.ml as ml


class SeqLoss(nn.Module):
    pass


class OSizeNetwork(nn.Module):
    def __init__(self):
        super(OSizeNetwork, self).__init__()
        self.hidden_dim = 100
        self.i2h = nn.Linear(2 + self.hidden_dim, self.hidden_dim)
        self.input_fc = nn.Linear(2 + self.hidden_dim, 100)
        self.output_size_fc = nn.Linear(100, 2)

    def forward(self, x, hidden_state):
        input_combined = torch.cat((x, hidden_state), dim=1)

        # Computing the next hidden state
        hidden = F.relu(self.i2h(input_combined))

        # Computing the output
        output = F.relu(self.input_fc(input_combined))
        output = F.relu(self.output_size_fc(output))
        return output, hidden


class OSizeModel(ml.TrainerModel):

    def __init__(self):
        modelname = 'output_size_model'
        model = OSizeNetwork()
        opti = optim.Adam(model.parameters(), lr=5e-3)
        crit = nn.MSELoss()
        visboard = ml.VisBoard(lib.create_identifier(modelname))
        super().__init__(model_name=modelname,
                         model=model,
                         opti=opti,
                         crit=crit,
                         visboard=visboard)

    def init_hidden(self):
        return torch.zeros(BATCH_SIZE, self.model.hidden_dim)


def encode_outputsize(im: lib.ImStack) -> Tuple[np.ndarray, np.ndarray]:
    x_shape = im.get_ndarray().shape[1:3]
    y_shape = im.semseg_masks[0].get_ndarray().shape[:2]

    x = np.array([x_shape[0], x_shape[1]]).astype(np.float32)
    y = np.array([y_shape[0], y_shape[1]]).astype(np.float32)
    return x, y


def o_size_preprocessor(s: lib.Subject, mode: ml.ModelMode):
    train_examples: List[lib.ImStack] = []
    test_examples: List[lib.ImStack] = []
    # noinspection PyTypeChecker
    for imstack in s.ims:
        if imstack.extra_info['purpose'] == 'train':
            train_examples.append(imstack)
        else:
            test_examples.append(imstack)
    assert (len(train_examples) > 0 and len(test_examples) > 0), f"Something is wrong with {s.name}"
    # print(train_examples)
    res = [encode_outputsize(im) for im in train_examples]

    test_example = random.choice(test_examples)
    res.append(encode_outputsize(test_example))
    return res


if __name__ == '__main__':
    # lib.reset_database()
    # sd = demo_data.SourceData('D:\\')
    # sd.build_arc(lib.Session())

    BATCH_SIZE = 4
    EPOCHS = 5

    output_size_train = ml.InMemoryDataset('arc', 'training', o_size_preprocessor, mode=ml.ModelMode.Train)

    train_loader = output_size_train.get_torch_dataloader(batch_size=BATCH_SIZE)

    net = OSizeModel()

    for epoch in range(EPOCHS):
        net.run_epoch(train_loader, epoch=epoch, batch_size=BATCH_SIZE)
