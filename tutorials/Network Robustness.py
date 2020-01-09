import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from trainer.ml.torch_utils import instantiate_model, TorchModel, fgsm_attack, train_model, load_testset, TestSet


class ConvNet(TorchModel):
    def __init__(self):
        super(ConvNet, self).__init__(model_name="convnet")
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class FCNet(TorchModel):
    def __init__(self):
        super(FCNet, self).__init__(model_name='fcnet')
        self.fc1 = nn.Linear(28 * 28, 28 * 14)
        self.fc2 = nn.Linear(28 * 14, 28 * 8)
        self.output = nn.Linear(28 * 8, 10)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return F.log_softmax(x, dim=1)


def test(model: TorchModel, device: torch.device, test_loader: torch.utils.data.DataLoader, epsilon: float, writer):
    wrong, correct, adv_success = 0, 0, 0
    adv_examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        data.requires_grad = True  # This enables a gradient based attack such as fgsm

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            wrong += 1
        else:
            # Prediction was correct, try to fool the network
            correct += 1

            loss = F.nll_loss(output, target)

            model.zero_grad()

            loss.backward()

            data_grad = data.grad.data

            perturbed_data = fgsm_attack(data, epsilon, data_grad)

            output = model(perturbed_data)

            adversarial_pred = output.max(1, keepdim=True)[1]  # max returns a tuple (values, indices)
            if np.random.rand() > 0.9 or True:
                perturbed_arr = perturbed_data.detach().cpu().numpy()
                writer.add_image(f'Adversarial Example with Epsilon={epsilon}', perturbed_arr[0])
                adv_examples.append((target.item(), adversarial_pred.item(), perturbed_arr))

            if adversarial_pred.item() != target.item():
                adv_success += 1

    print(f'Correct: {correct}; Wrong: {wrong}')
    print(f'Adversarial Successes: {adv_success}')
    acc = correct / (wrong + correct)
    return acc, adv_examples


def run_adv_attacks(model):
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    for epsilon in epsilons:
        acc, meta = test(model, device, test_loader, epsilon=epsilon)
        true_target, adv_pred, adv_ex = random.choice(meta)
        plt.title(f"Epsilon: {epsilon}, True Target: {true_target}, Pred: {adv_pred}")
        plt.imshow(adv_ex.squeeze())
        plt.show()
        print(f"Accuracy for epsilon={epsilon}: {acc}")


if __name__ == '__main__':
    train_loader, test_loader = load_testset(TestSet.MNIST)

    fc_net, fc_writer = instantiate_model(FCNet, weights_path='fc_net.pt', data_loader=train_loader)
    conv_net, conv_writer = instantiate_model(ConvNet, weights_path='conv_net.pt', data_loader=train_loader)

    from trainer.ml.torch_utils import visualize_input_batch, visualize_model_weights

    visualize_input_batch(train_loader, fc_writer)

    if not os.path.exists('fc_net.pt'):
        fc_state = train_model(train_loader, fc_net, fc_writer)
        torch.save(fc_state, 'fc_net.pt')

    if not os.path.exists('conv_net.pt'):
        conv_state = train_model(train_loader, conv_net, conv_writer)
        torch.save(conv_state, 'conv_net.pt')

    # Visualize model weights
    # visualize_model_weights(fc_net, fc_writer)
    visualize_model_weights(conv_net, conv_writer)