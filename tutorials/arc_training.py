import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import trainer.lib as lib
import trainer.ml as ml


class NumberToEncoding(nn.Module):
    def __init__(self):
        super(NumberToEncoding, self).__init__()
        self.n = 3

    def forward(self, numbers):
        # assert (len(numbers.size()) == 2)
        # ml.logger.log(str(numbers))
        return numbers


class EncodingToNumber(nn.Module):
    def __init__(self, n: int):
        super(EncodingToNumber, self).__init__()
        self.n = n
        # noinspection PyArgumentList
        self.bin_decoder = torch.Tensor([x ** 2 for x in range(1, n + 1)]).to(ml.torch_device)

    def forward(self, encoding):
        encoding = encoding[0]  # Assuming batch size 1!
        # encoding = encoding.round()
        cuts = [encoding[number_i * self.n: (number_i + 1) * self.n] for number_i in range(2)]
        dots = [torch.dot(t, self.bin_decoder) for t in cuts]
        return torch.stack(dots).unsqueeze(0)


class OSizeNetwork(nn.Module):
    def __init__(self):
        super(OSizeNetwork, self).__init__()
        self.hidden_depth = 30
        self.layer_dim = 100
        # self.i2h = nn.Linear(2 + self.hidden_dim, self.hidden_dim)
        self.encoding_layer = NumberToEncoding()
        # self.decoding_layer = EncodingToNumber(5)
        self.decoding_layer = nn.LogSoftmax(dim=1)

        # self.rnn_cell = nn.GRUCell(2, self.hidden_dim)
        self.rnn_cell = ml.ConvGRUCell(
            input_size=(30, 30),
            input_dim=11,
            hidden_dim=self.hidden_depth,
            kernel_size=(3, 3),
            bias=True,
            dtype=torch.FloatTensor
        )

    def forward(self, inps: List, hidden_states: List):
        gridsize, hidden_state = inps[0], hidden_states[0]

        x = self.encoding_layer(gridsize)

        # Computing the next hidden state
        hidden_state = self.rnn_cell(x, hidden_state)

        # input_combined = torch.cat((x, hidden_state), dim=1)
        input_combined = hidden_state
        # Computing the output
        output = torch.relu(self.input_fc(input_combined))
        output = torch.relu(self.hidden_fc(output))
        output = torch.relu(self.output_size_fc(output))

        # output = torch.cat([torch.mul(output, m) for m in [1., 2., 3., 4., 5.]], dim=1)

        output = torch.relu(self.reduction_layer(output))
        # output = self.decoding_layer(output)
        return output, [hidden_state]


class OSizeModel(ml.TrainerModel):

    def __init__(self):
        modelname = 'output_size_model'
        model = OSizeNetwork()
        opti = optim.Adam(
            model.parameters(),
            lr=5e-3
        )
        crit = nn.MSELoss()
        super().__init__(model_name=modelname,
                         model=model,
                         opti=opti,
                         crit=crit)

    def init_hidden(self) -> List[torch.Tensor]:
        first = [torch.zeros(BATCH_SIZE, self.model.hidden_dim).to(ml.torch_device)]
        return first

    def handle_minibatch(self, seq: List[Tuple[List[torch.Tensor], torch.Tensor]], metric: ml.TrainerMetric = None):
        attempts = 0
        loss = self.train_on_minibatch(seq, evaluator=metric)
        while loss > 1.0 and attempts < 10:
            loss = self.train_on_minibatch(seq, evaluator=metric)
            attempts += 1
        return loss


def encode_outputsize(im: lib.ImStack) -> Tuple[List[np.ndarray], np.ndarray]:
    x_shape = im.get_ndarray().shape[1:3]
    y_shape = im.semseg_masks[0].get_ndarray().shape[:2]

    x = np.array([x_shape[0], x_shape[1]]).astype(np.float32)
    y = np.array([y_shape[0], y_shape[1]]).astype(np.float32)
    return [x], y


def extract_train_test(s: lib.Subject):
    train_examples: List[lib.ImStack] = []
    test_examples: List[lib.ImStack] = []
    # noinspection PyTypeChecker
    for imstack in s.ims:
        if imstack.extra_info['purpose'] == 'train':
            train_examples.append(imstack)
        else:
            test_examples.append(imstack)
    assert (len(train_examples) > 0 and len(test_examples) > 0), f"Something is wrong with {s.name}"
    return train_examples, test_examples


def o_size_preprocessor(s: lib.Subject, mode: ml.ModelMode):
    train_examples, test_examples = extract_train_test(s)
    # print(train_examples)
    res = [encode_outputsize(im) for im in train_examples]

    test_example = random.choice(test_examples)
    res.append(encode_outputsize(test_example))
    return res


def encode_depthmap(im: lib.ImStack, n_classes=11, max_grid=30) -> Tuple[List[np.ndarray], np.ndarray]:
    x, y = im.get_ndarray()[0, :, :, 0], im.semseg_masks[0].get_ndarray()[:, :, 0]
    inp = np.zeros((max_grid, max_grid, n_classes), dtype=np.float32)
    trgt = np.zeros((max_grid, max_grid, 2), dtype=np.float32)

    foreground_x, foreground_y = np.ones_like(x, dtype=np.float32), np.ones_like(y, dtype=np.float32)

    inp[foreground_x.shape[0]:, :, 0] = 1.
    inp[:, foreground_x.shape[1]:, 0] = 1.
    for w in range(x.shape[0]):
        for h in range(x.shape[1]):
            c = x[w, h] + 1
            inp[w, h, c] = 1.

    trgt[foreground_y.shape[0]:, :, 0] = 1.
    trgt[:, foreground_y.shape[1]:, 0] = 1.
    trgt[:foreground_y.shape[0], :foreground_y.shape[1], 1] = 1.

    return [inp], trgt


def depthmap_preprocessor(s: lib.Subject, mode: ml.ModelMode):
    train_examples, test_examples = extract_train_test(s)
    res = [encode_depthmap(im) for im in train_examples]

    test_example = random.choice(test_examples)
    res.append(encode_depthmap(test_example))
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

    output_size_train = ml.InMemoryDataset('arc', 'training', depthmap_preprocessor, mode=ml.ModelMode.Train)
    output_size_test = ml.InMemoryDataset('arc', 'test', depthmap_preprocessor, mode=ml.ModelMode.Eval)

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
