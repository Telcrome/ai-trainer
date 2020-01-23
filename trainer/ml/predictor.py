from torchvision.models.vgg import vgg19

if __name__ == '__main__':
    net = vgg19(pretrained=True)
    print(net)
