import argparse
from model import Generator, Discriminator
import torch
import torch.nn as nn
import torch.optim as optim
from data import Celeba_Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def train(args):
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            #nn.init.xavier_uniform_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

    netG = Generator(args).to(device)
    netD = Discriminator(args).to(device)
    if(device.type == 'cuda') and (args.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(args.ngpu)))
        netD = nn.DataParallel(netD, list(range(args.ngpu)))
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, 0.999))

    face_dataset = Celeba_Dataset(imgfolder=args.data_folder,
                                  transforms=transforms.Compose([
                                      transforms.Resize(args.image_size),
                                      transforms.CenterCrop(args.image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ]))

    train_loader = DataLoader(dataset=face_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers)

    for epoch in range(args.epoch):
        D_losses = []
        G_losses = []
        for i, data in enumerate(train_loader):

            netD.zero_grad()
            real_img = data.to(device)
            print(real_img.shape)
            b_size = real_img.size(0)
            real_label = torch.ones((b_size), device=device)
            real_output = netD(real_img).view(-1)
            errD_real = criterion(real_output, real_label)
            errD_real.backward()

            noise = torch.randn(b_size, args.length_z, 1, 1, device=device)
            print(noise.shape)
            fake_img = netG(noise)
            fake_label = torch.zeros((b_size), device=device)
            fake_output = netD(fake_img.detach()).view(-1)
            errD_fake = criterion(fake_output, fake_label)
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            output = netD(fake_img).view(-1)
            errG = criterion(output, real_label)
            errG.backward()
            optimizerG.step()

            D_losses.append(errD.item())
            G_losses.append(errG.item())

            
            break
        print()
        break


def test(args):
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # run setting
    parser.add_argument('--opt', type=str, default="test")

    # data setting
    parser.add_argument('--data_folder', type=str, default='../_data/img_align_celeba/')
    parser.add_argument('--val_data', type=str, default='../_data/img_align_celeba/')
    parser.add_argument('--image_size', type=int, nargs='+', default=64)

    # ckpt setting
    parser.add_argument('--save_checkpoint_interval', type=int, default=20)

    # network setting
    parser.add_argument('--length_z', type=int, default=100)

    # training setting
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--workers', type=int, default=2)

    # gpu setting
    parser.add_argument('--ngpu', type=int, default=1)

    args = parser.parse_args()

    if(args.opt == "train"):
        train(args)
    elif(args.opt == "test"):
        test(args)
