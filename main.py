import argparse

from model import Generator, Discriminator
from data import Celeba_Dataset

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.utils as vutils

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

    fixed_noise = torch.randn(args.save_img, args.length_z, 1, 1, device=device)

    for epoch in range(args.epoch):
        D_losses = []
        G_losses = []
        for i, data in enumerate(train_loader):
            netD.zero_grad()
            real_img = data.to(device)
            b_size = real_img.size(0)
            real_label = torch.ones((b_size), device=device)
            real_output = netD(real_img).view(-1)
            errD_real = criterion(real_output, real_label)
            errD_real.backward()

            noise = torch.randn(b_size, args.length_z, 1, 1, device=device)
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

            if(i%100 == 0):
                print(errD.item(), errG.item())
                with torch.no_grad():
                    fake_img = netG(fixed_noise)
                    for img_no in range(args.save_img):
                        vutils.save_image(fake_img[img_no], args.save_img_folder + str(epoch) + "_" + str(i) + "_" + str(img_no) + ".jpg")

        print(str(epoch) + " epoch\n")
        print("\tD loss: " + str(np.mean(np.array(D_losses))))
        print("\tG loss: " + str(np.mean(np.array(G_losses))))

        with torch.no_grad():
            fake_img = netG(fixed_noise)
            for img_no in range(args.save_img):
                vutils.save_image(fake_img[img_no], args.save_img_folder + str(epoch) + "_fin_" + str(img_no) + ".jpg")

        if(epoch % args.save_checkpoint_interval == 0):
            torch.save({
                "Generator_state_dict": netG.state_dict(),
                "Discriminator_state_dict": netD.state_dict(),
            }, args.checkpoint_path + str(epoch) + "_checkpoint.pth.tar")

def test(args):
    checkpoint = torch.load(args.checkpoint)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    netG = Generator(args)
    netG.load_state_dict(checkpoint['Generator_state_dict'])
    netG = netG.to(device)

    fixed_noise = torch.randn(args.save_img, args.length_z, 1, 1, device=device)
    fake_img = netG(fixed_noise)
    for img_no in range(args.save_img):
        vutils.save_image(fake_img[img_no], args.save_img_folder + str(img_no) + ".jpg")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # run setting
    parser.add_argument('--opt', type=str, default="test")

    # data setting
    parser.add_argument('--data_folder', type=str, default='./_data/img_align_celeba/')
    parser.add_argument('--image_size', type=int, nargs='+', default=64)

    # ckpt setting
    parser.add_argument('--save_checkpoint_interval', type=int, default=2)
    parser.add_argument('--checkpoint_path', type=str, default="./_model/")
    parser.add_argument('--checkpoint', type=str, default="./_model/checkpoint.pth.tar")

    # network setting
    parser.add_argument('--length_z', type=int, default=100)

    # training setting
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--save_img', type=int, default=10)
    parser.add_argument('--save_img_folder', type=str, default="./_output/")

    # gpu setting
    parser.add_argument('--ngpu', type=int, default=1)

    args = parser.parse_args()

    if(args.opt == "train"):
        train(args)
    elif(args.opt == "test"):
        test(args)
