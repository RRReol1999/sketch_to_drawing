import argparse
import os
from random import random

import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.append('/content/drive/MyDrive/StyTR-2-main/models')
import transformer as transformer
import StyTR as StyTR
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, content_root, style_root, transform):
        super(FlatFolderDataset, self).__init__()
        self.transform = transform
        self.content_paths = []
        self.style_paths = []

        style_filenames = os.listdir(style_root)
        for filename in style_filenames:
            content_filename = filename.replace('_paint', '')
            style_path = os.path.join(style_root, filename)
            content_path = os.path.join(content_root, content_filename)
            if os.path.exists(style_path):
                self.content_paths.append(content_path)
                self.style_paths.append(style_path)

    def __getitem__(self, index):
        content_path = self.content_paths[index]
        style_path = self.style_paths[index]
        seed = torch.randint(0, 2 ** 32, (1,)).item()
        random.seed(seed)
        torch.manual_seed(seed)
        content_img = Image.open(content_path).convert('RGB')
        content_img = self.transform(content_img)
        random.seed(seed)
        torch.manual_seed(seed)
        style_img = Image.open(style_path).convert('RGB')
        style_img = self.transform(style_img)

        return content_img, style_img

    def __len__(self):
        return len(self.content_paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', default='./drive/MyDrive/StyTR-2-main/input/content', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='./drive/MyDrive/StyTR-2-main/input/sty', type=str,
                    # wikiart dataset crawled from https://www.wikiart.org/
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str,
                    default='./drive/MyDrive/StyTR-2-main/experiments/vgg_normalised.pth')  # run the train.py, please download the pretrained vgg checkpoint

# training options
parser.add_argument('--save_dir', default='./drive/MyDrive/StyTR-2-main/input/test',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./drive/MyDrive/StyTR-2-main/logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=9.0)
parser.add_argument('--content_weight', type=float, default=7.5)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
print(torch.cuda.current_device());
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

vgg = StyTR.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = StyTR.decoder
embedding = StyTR.PatchEmbed()

Trans = transformer.Transformer()
with torch.no_grad():
    network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)
network.train()

network.to(device)
content_tf = train_transform()
style_tf = train_transform()

dataset = FlatFolderDataset(args.content_dir, args.style_dir, content_tf)
data_loader = iter(data.DataLoader(
    dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam([
    {'params': network.transformer.parameters()},
    {'params': network.decode.parameters()},
    {'params': network.embedding.parameters()},
], lr=args.lr)

if not os.path.exists(args.save_dir + "/test"):
    os.makedirs(args.save_dir + "/test")

for i in tqdm(range(args.max_iter)):

    if i < 1e4:
        warmup_learning_rate(optimizer, iteration_count=i)
    else:
        adjust_learning_rate(optimizer, iteration_count=i)

    # print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))
    content_images, style_images = next(data_loader)
    content_images = content_images.to(device)
    style_images = style_images.to(device)
    out, loss_c, loss_s, l_identity1, l_identity2 = network(content_images, style_images)

    if i % 100 == 0:
        output_name = '{:s}/test/{:s}{:s}'.format(
            args.save_dir, str(i), ".jpg"
        )
        out = torch.cat((content_images, out), 0)
        out = torch.cat((style_images, out), 0)
        save_image(out, output_name)

    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1)

    print(loss.sum().cpu().detach().numpy(), "-content:", loss_c.sum().cpu().detach().numpy(), "-style:",
          loss_s.sum().cpu().detach().numpy()
          , "-l1:", l_identity1.sum().cpu().detach().numpy(), "-l2:", l_identity2.sum().cpu().detach().numpy()
          )

    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
    writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
    writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
    writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
    writer.add_scalar('total_loss', loss.sum().item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.transformer.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

        state_dict = network.decode.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                       i + 1))
        state_dict = network.embedding.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/embedding_iter_{:d}.pth'.format(args.save_dir,
                                                         i + 1))

writer.close()


