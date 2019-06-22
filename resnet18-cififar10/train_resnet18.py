# coding=utf-8
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from resnet import ResNet18
from torch import nn,optim
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument(
    '--outf',
    default='./model/',
    help='folder to output images and model checkpoints')  # 输出结果保存路径
parser.add_argument(
    '--net',
    default='./model/Resnet18.pth',
    help="path to net (to continue training)")  # 恢复训练时的模型路径
args = parser.parse_args()


def main():

    batchsz = 256

    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]), download=True)

    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    print(model)
    for epoch in range(200):
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #
        print(epoch, 'loss:', loss.item())

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # [b, 3, 32, 32]
                # [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)

            acc = total_correct / total_num
            print(epoch, 'acc:', acc)


if __name__ == '__main__':
    main()