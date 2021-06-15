import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import jittor as jt
from jittor import nn
from jittor import transform
from jittor.dataset import Dataset
from jittor.lr_scheduler import CosineAnnealingLR

from jimm import resnext101_32x8d
jt.flags.use_cuda = 0

def get_img(path):
    return Image.open(path).convert('RGB')


def get_train_transforms():
    return transform.Compose([
        transform.RandomCropAndResize((448, 448)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.ImageNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


def get_valid_transforms():
    return transform.Compose([
        transform.Resize(512),
        transform.CenterCrop(448),
        transform.ToTensor(),
        transform.ImageNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


class CUB200(Dataset):
    def __init__(self, root_dir, batch_size=16, part='train', shuffle=False, transform=None):
        super(CUB200, self).__init__()
        self.root_dir = root_dir
        self.part = part
        self.transform = transform
        self.image_list = []
        self.id_list = []
        image_path = os.path.join(root_dir, 'images')
        id2name = np.genfromtxt(os.path.join(root_dir, 'images.txt'), dtype=str)
        id2train = np.genfromtxt(os.path.join(root_dir, 'train_test_split.txt'), dtype=int)
        target = 1 if part == 'train' else 0
        for i in range(id2name.shape[0]):
            if id2train[i, 1] != target:
                continue
            label = int(id2name[i, 1][:3]) - 1
            self.image_list.append(os.path.join(image_path, id2name[i, 1]))
            self.id_list.append(label)
        self.set_attrs(
            batch_size=batch_size,
            total_len=len(self.id_list),
            shuffle=shuffle
        )

    def __getitem__(self, index):
        label = self.id_list[index]
        image_path = os.path.join(self.root_dir, self.image_list[index])
        image = get_img(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, accum_iter, scheduler):
    model.train()
    total_acc = 0
    total_num = 0
    losses = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    for i, (images, labels) in enumerate(pbar):
        print(images.shape)
        output = model(images)
        loss = criterion(output, labels)

        optimizer.backward(loss)
        if (i + 1) % accum_iter == 0 or i + 1 == len(train_loader):
            optimizer.step(loss)

        # print(output)
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]
        losses.append(loss.data[0])

        pbar.set_description(f'Epoch {epoch} loss={sum(losses) / len(losses):.2f}'
                             f'acc={total_acc / total_num:.2f}')
    scheduler.step()


def valid_one_epoch(model, val_loader, epoch):
    model.eval()
    total_acc = 0
    total_num = 0

    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [VALID]')
    for images, labels in val_loader:
        output = model(images)
        pred = np.argmax(output.data, axis=1)

        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]

        pbar.set_description(f'Epoch {epoch} acc={total_acc / total_num:.2f}')

    acc = total_acc / total_num
    return acc


if __name__ == '__main__':
    jt.set_global_seed(648)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eta_min', type=float, default=1e-5)
    parser.add_argument('--T_max', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--accum_iter', type=int, default=1)
    args = parser.parse_args()
    options = {
        'root_dir': '/dataset/CUB2011/CUB_200_2011/',
        'num_classes': 200,
        'threshold': 0.8,
        'lr': args.lr,
        'eta_min': args.eta_min,
        'T_max': args.T_max,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'accum_iter': args.accum_iter,
    }

    train_loader = CUB200(options['root_dir'], options['batch_size'], 'train', shuffle=True,
                          transform=get_train_transforms())
    val_loader = CUB200(options['root_dir'], options['batch_size'], 'valid', shuffle=False,
                        transform=get_valid_transforms())

    model = resnext101_32x8d(pretrained=True, num_classes=options['num_classes'])
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.parameters(), options['lr'], weight_decay=options['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, options['T_max'], options['eta_min'])

    best_acc = options['threshold']
    for epoch in range(options['epochs']):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, options['accum_iter'], scheduler)
        acc = valid_one_epoch(model, val_loader, epoch)
        if acc > best_acc:
            best_acc = acc
            model.save(f'resnext101_32x8d-{epoch}-{acc:.2f}.pkl')
