import argparse

def train():


    return 1

def test():

    return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # run setting
    parser.add_argument('--opt', type=str, default="test")

    # data setting
    parser.add_argument('--train_data', type=str, default='./_data/train_data/')
    parser.add_argument('--val_data', type=str, default='./_data/val_data/')
    parser.add_argument('--image_size', type=int, nargs='+', default=[64, 64, 3])

    # ckpt setting
    parser.add_argument('--save_checkpoint_interval', type=int, default=20)

    # network setting
    parser.add_argument('--length_z', type=int, default=100)

    # training setting
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--workers', type=int, default=2)

    # gpu setting
    parser.add_argument('--ngpu', type=int, default=1)

    args = parser.parse_args()

    if(args.opt == "train"):
        train()
    elif(args.opt == "test"):
        test()
