import torch
import torch.nn as nn
import torch.optim as optim

import trainer.lib as lib
import trainer.ml as ml


class SeqLoss(nn.Module):
    pass


class OSizeNetwork(nn.Module):
    def __init__(self):
        super(OSizeNetwork, self).__init__()
        self.hidden_dim = 10
        self.rnn = nn.RNN(2, self.hidden_dim, num_layers=2, batch_first=True)
        self.output_size_fc = nn.Linear(10, 2)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = torch.zeros((2, batch_size, self.hidden_dim))

        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        return self.output_size_fc(out), hidden


class OSizeModel(ml.TrainerModel):

    def __init__(self):
        modelname = 'output_size_model'
        model = OSizeNetwork()
        opti = optim.Adam(model.parameters(), lr=5e-3)
        crit = nn.CrossEntropyLoss()
        visboard = ml.VisBoard(lib.create_identifier(modelname))
        super().__init__(model_name=modelname,
                         model=model,
                         opti=opti,
                         crit=crit,
                         visboard=visboard)


def o_size_preprocessor(s: lib.Subject, mode: ml.ModelMode):
    train_examples = []
    test_example = None
    # noinspection PyTypeChecker
    for imstack in s.ims:
        if imstack.extra_info['purpose'] == 'train':
            train_examples.append(imstack)
        else:
            assert (test_example is None), "There can only be one test example"
            test_example = imstack
    assert (len(train_examples) > 0 and test_example is not None), f"Something is wrong with {s.name}"
    print(train_examples)


if __name__ == '__main__':
    # lib.reset_database()
    # sd = demo_data.SourceData('D:\\')
    # sd.build_arc(lib.Session())

    BATCH_SIZE = 32
    EPOCHS = 5

    output_size_train = ml.InMemoryDataset('arc', 'training', o_size_preprocessor, mode=ml.ModelMode.Train)

    train_loader = output_size_train.get_torch_dataloader(batch_size=32)

    net = OSizeModel()

    for epoch in range(EPOCHS):
        net.run_epoch(train_loader, epoch=epoch, batch_size=BATCH_SIZE)
