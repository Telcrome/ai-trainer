import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import trainer.lib as lib
import trainer.ml as ml


class OSizeNetwork(nn.Module):
    def __init__(self):
        super(OSizeNetwork, self).__init__()
        self.hidden_dim = 100
        self.layer_dim = 100
        # self.i2h = nn.Linear(2 + self.hidden_dim, self.hidden_dim)
        self.rnn_cell = nn.GRUCell(2, self.hidden_dim)
        self.input_fc = nn.Linear(self.hidden_dim, self.layer_dim)
        self.hidden_fc = nn.Linear(self.layer_dim, self.layer_dim)
        self.output_size_fc = nn.Linear(self.layer_dim, 10)
        self.reduction_layer = nn.Linear(10, 2)

    def forward(self, inps, hidden_states):
        x, hidden_state = inps[0], hidden_states[0]
        # Computing the next hidden state
        hidden_state = self.rnn_cell(x, hidden_state)

        # input_combined = torch.cat((x, hidden_state), dim=1)
        input_combined = hidden_state
        # Computing the output
        output = F.relu(self.input_fc(input_combined))
        output = F.relu(self.hidden_fc(output))
        output = F.relu(self.output_size_fc(output))

        # output = torch.cat([torch.mul(output, m) for m in [1., 2., 3., 4., 5.]], dim=1)

        output = F.relu(self.reduction_layer(output))
        return output, hidden_state


class OSizeModel(ml.TrainerModel):

    def __init__(self):
        modelname = 'output_size_model'
        model = OSizeNetwork()
        opti = optim.Adam(
            model.parameters(),
            lr=5e-3,
            weight_decay=0.2
        )
        crit = nn.MSELoss()
        super().__init__(model_name=modelname,
                         model=model,
                         opti=opti,
                         crit=crit)

    def init_hidden(self) -> List[torch.Tensor]:
        return [torch.zeros(BATCH_SIZE, self.model.hidden_dim).to(ml.torch_device)]

    def handle_minibatch(self, seq: List[Tuple[List[torch.Tensor], torch.Tensor]], metric: ml.TrainerMetric = None):
        attempts = 0
        loss = self.train_on_minibatch(seq, evaluator=metric)
        while loss > 1.0 and attempts < 50:
            loss = self.train_on_minibatch(seq, evaluator=metric)
            attempts += 1
        return loss


def encode_outputsize(im: lib.ImStack) -> Tuple[List[np.ndarray], np.ndarray]:
    x_shape = im.get_ndarray().shape[1:3]
    y_shape = im.semseg_masks[0].get_ndarray().shape[:2]

    x = np.array([x_shape[0], x_shape[1]]).astype(np.float32)
    y = np.array([y_shape[0], y_shape[1]]).astype(np.float32)
    return [x], y


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


class ARCMetric(ml.TrainerMetric):

    def __init__(self):
        self.correct = 0
        self.wrong = 0

    def update(self, prediction: np.ndarray, target: np.ndarray):
        prediction = np.round(prediction)
        for b_id in range(prediction.shape[0]):
            if prediction[b_id, 0] == target[b_id, 0] and prediction[b_id, 1] == target[b_id, 1]:
                self.correct += 1
            else:
                self.wrong += 1

    def get_result(self):
        return self.correct / (self.correct + self.wrong)


if __name__ == '__main__':
    # lib.reset_database()
    # sd = demo_data.SourceData('D:\\')
    # sd.build_arc(lib.Session())

    BATCH_SIZE = 1
    EPOCHS = 5

    output_size_train = ml.InMemoryDataset('arc', 'training', o_size_preprocessor, mode=ml.ModelMode.Train)
    output_size_test = ml.InMemoryDataset('arc', 'test', o_size_preprocessor, mode=ml.ModelMode.Eval)

    train_loader = output_size_train.get_torch_dataloader(batch_size=BATCH_SIZE)
    test_loader = output_size_train.get_torch_dataloader(batch_size=BATCH_SIZE)

    net = OSizeModel()

    for epoch in range(EPOCHS):
        metric = ARCMetric()
        net.run_epoch(train_loader, epoch=epoch, batch_size=BATCH_SIZE, metric=metric)

        # Testing the performance on the test set
        metric = ARCMetric()
        net.evaluate(test_loader, metric)
        # print(f'Test Output size accuracy: {metric.get_result()}')
        # device = ml.torch_device
        # with torch.no_grad():
        #     x = torch.Tensor([[3, 5]]).to(device)
        #     h = torch.zeros((1, 100)).to(device)
        #     y, h = net.model(x, h)
        # print(y)
