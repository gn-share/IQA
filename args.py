import argparse
import os
import random
import torch
import time

# main_path = ".."
main_path = os.environ['HOME']


os.environ['OMP_NUM_THREADS'] = '1'

folder_path = {
    'live': main_path + '/image_data/LIVE/',  #
    'csiq': main_path + '/image_data/CSIQ/',  #
    'tid2013': main_path + '/image_data/tid2013',
    'livec': main_path + '/image_data/ChallengeDB_release/',  #
    'koniq': main_path + '/image_data/koniq/',  #
    'bid': main_path + '/image_data/BID/',  #
}

img_num = {
    'live': list(range(0, 29)),
    'csiq': list(range(0, 30)),
    'tid2013': list(range(0, 25)),
    'livec': list(range(0, 1162)),
    'koniq': list(range(0, 10073)),
    'bid': list(range(0, 586)),
}


num_workers = {
    'live': 3,
    'csiq': 5,
    'tid2013': 5,
    'livec': 3,
    'koniq': 3,
    'bid': 8,
}


def try_gpu(i=0):  # @save
    """如果存在,则返回gpu(i),否则退出"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    #  The i-th cuda number does not exist
    print(f"The {i}-th cuda index does not exist !!!")
    exit(0)


def Configs():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset', dest='dataset', type=str, default='livec',
                        help='Support datasets: livec|koniq|bid|live|csiq|tid2013')

    # train test patch_num
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int,
                        default=50, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int,
                        default=50, help='Number of sample patches from testing image')

    # learn rate
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-5, help='Learning rate')
    parser.add_argument('--lrratio', dest='lrratio', type=int,
                        default=2, help='Learning rate ratio')
    parser.add_argument('--weight_decay', dest='weight_decay',
                        type=float, default=5e-4, help='Weight decay')

    # batch size
    parser.add_argument('--batch_size', dest='batch_size',
                        type=int, default=48, help='Batch size')
    parser.add_argument('--test_batch_size', dest='test_batch_size',
                        type=int, default=48, help='The test batch size')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=5, help='Epochs for training')

    # imgae setting
    parser.add_argument('--patch_size', dest='patch_size', type=int,
                        default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num',
                        type=int, default=10, help='Train-test times')

    # seed
    parser.add_argument('--use_seed', default=False,
                        action='store_true', help="Is it use the seed")
    parser.add_argument('--seed', dest='seed',
                        type=int, default=0, help='Choose the seed number')
    parser.add_argument('--random_single', default=False,
                        action="store_true", help="Is it only random once")

    # choose device to computer
    parser.add_argument('--use_cpu', default=False,
                        action='store_true', help="Is it use the cpu to computer")
    parser.add_argument('--cudas', dest='cudas', type=str,
                        default='0', help='Choose the cuda number')

    # Is it to test the code
    parser.add_argument('--test', default=False,
                        action='store_true', help=" Is it to test the code")

    # Is it to use the tqdm
    parser.add_argument('--use_tqdm', default=False,
                        action='store_true', help="Is it to use the tqdm")

    config = parser.parse_args()
    if config.use_seed:
        random.seed(config.seed)

    # choose num_workers
    config.num_workers = num_workers[config.dataset] if config.dataset in num_workers.keys(
    ) else 3

    # Choose whether to test model
    if config.test:
        print("Start to test model")
        config.printf = print
        config.train_test_num = 1
        config.epochs = 30
        config.use_tqdm = True
    else:
        # write the output to file
        result_file = f"./results/{config.dataset}-{time.strftime('%d-%H_%M', time.localtime())}.txt"
        if not os.path.exists("results"):
            os.mkdir("results")

        # write to file
        def printf(s, file=result_file):
            with open(file, "a+") as f:
                print(s, file=f)
        config.printf = printf

    # Isn't using gpu
    if config.use_cpu:
        config.device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.cudas
        config.device = try_gpu(i=0)
        config.devices = []
        # devices save more gpu device index.
        for cuda in range(len(config.cudas.split(","))):
            config.devices.append(try_gpu(i=cuda))

    config.printf(config)

    config.folder_path = folder_path
    config.img_num = img_num
    return config
