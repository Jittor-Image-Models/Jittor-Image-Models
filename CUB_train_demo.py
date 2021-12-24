import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
import jittor as jt
from jittor import nn
from jittor import transform
from jittor.dataset import Dataset
from jittor.lr_scheduler import CosineAnnealingLR
from jimm.data import RandomMixup, RandomCutmix, RandomChoice
from jimm.loss import CrossEntropy, LabelSmoothingCrossEntropy

from jimm import resnext101_32x8d, resnet50, tf_efficientnet_b4_ns, vit_tiny_patch16_224
jt.flags.use_cuda = 1

def get_img(path):
    return Image.open(path).convert('RGB')


def get_train_transforms():
    return transform.Compose([
        transform.RandomCropAndResize((224, 224)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.ImageNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


def get_valid_transforms():
    return transform.Compose([
        transform.Resize(256),
        transform.CenterCrop(224),
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
        self.collate = RandomChoice([RandomCutmix(options['num_classes'], p=1.0, alpha=1.0), 
                                    RandomMixup(options['num_classes'], p=1.0, alpha=0.2)])
        self.set_attrs(
            batch_size=batch_size,
            total_len=len(self.id_list),
            shuffle=shuffle,
            num_workers=8,
        )

    def collate_batch(self, batch):
        batch = super().collate_batch(batch)
        if self.part == 'train':
            batch = self.collate(*batch)
        return batch

    def __getitem__(self, index):
        label = self.id_list[index]
        image_path = os.path.join(self.root_dir, self.image_list[index])
        image = get_img(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


@jt.enable_grad()
def train_one_epoch(model, train_loader, criterion, optimizer, epoch, accum_iter, scheduler):
    model.train()
    total_acc = 0
    total_num = 0
    losses = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    for i, (images, labels) in enumerate(pbar):
        output = model(images)
        if labels.requires_grad:
            labels.stop_grad()
        loss = criterion(output, labels)

        optimizer.backward(loss)
        if (i + 1) % accum_iter == 0 or i + 1 == len(train_loader):
            optimizer.step(loss)

        if labels.ndim == 2:
            labels = jt.argmax(labels, 1)[0]
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]
        losses.append(loss.data[0])

        pbar.set_description(f'Epoch {epoch} loss={sum(losses) / len(losses):.2f}'
                             f'acc={total_acc / total_num:.2f}')
    scheduler.step()


@jt.no_grad()
def valid_one_epoch(model, val_loader, epoch):
    model.eval()
    total_acc = 0
    total_num = 0

    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [VALID]')
    for images, labels in pbar:
        output = model(images)
        pred = np.argmax(output.data, axis=1)

        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]

        pbar.set_description(f'Epoch {epoch} acc={total_acc / total_num:.2f}')

    acc = total_acc / total_num
    logger.info(f' Epoch: {epoch} , Validation multi-class accuracy = {acc :.4f}')
    return acc


if __name__ == '__main__':
    jt.set_global_seed(648)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eta_min', type=float, default=1e-5)
    parser.add_argument('--T_max', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--log_info', type=str, default='Log')
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

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(args.log_info+".log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    train_loader = CUB200(options['root_dir'], options['batch_size'], 'train', shuffle=True,
                          transform=get_train_transforms())
    val_loader = CUB200(options['root_dir'], options['batch_size'], 'valid', shuffle=False,
                        transform=get_valid_transforms())

    # model = vit_tiny_patch16_224(pretrained=True, num_classes=options['num_classes'])
    model = resnet50(pretrained=True, num_classes=options['num_classes'])
    # criterion = CrossEntropy()
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    # criterion = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.parameters(), options['lr'], weight_decay=options['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, options['T_max'], options['eta_min'])

    best_acc = options['threshold']
    for epoch in range(options['epochs']):
        print(optimizer.lr)
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, options['accum_iter'], scheduler)
        acc = valid_one_epoch(model, val_loader, epoch)
        if acc > best_acc:
            best_acc = acc
            model.save(f'resnet50-cub.pkl')