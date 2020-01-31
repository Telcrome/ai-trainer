"""
Augmentations can artificially increase dataset size.

Augmenting using adversarial attacks can work as a method of regularization.
"""

import torch


def oneshot_attack(image: torch.Tensor, max_change_percentage: float, data_grad: torch.Tensor, data_range=(-1., 1.)):
    """
    Oneshot attack, takes one step against the gradient direction to increase the error.
    """
    # sign_data_grad = data_grad.sign()
    epsilon = (((max_change_percentage * image) / data_grad).sum() / image.numel()).item()
    perturbed_image = image + epsilon * data_grad
    perturbed_image = torch.clamp(perturbed_image, data_range[0], data_range[1])
    return perturbed_image


def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor, data_range=(-1., 1.)):
    """
    Fast gradient sign method attack as proposed by:
    https://arxiv.org/pdf/1712.07107.pdf

    FGSM is an attack for an infinity-norm bounded adversary.
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, data_range[0], data_range[1])
    return perturbed_image
