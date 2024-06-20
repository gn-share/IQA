import torch
from scipy import stats
import numpy as np
import model
import data_loader
import time
from tqdm import tqdm
import hyper_model


class BaseIQASolver(object):

    def __init__(self, config, path, train_idx, test_idx):
        self.print = config.printf
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.device = config.device
        self.lr = config.lr
        self.lrratio = config.lrratio
        self.use_tqdm = config.use_tqdm

        self.model = model.BaseModel(pretrained=True).to(self.device)
        self.l1_loss = torch.nn.L1Loss().to(self.device)

        backbone_params = list(map(id, self.model.res.parameters()))
        self.wave_params = filter(lambda p: id(
            p) not in backbone_params, self.model.parameters())

        self.weight_decay = config.weight_decay
        paras = [{'params': self.wave_params, "lr": self.lr*self.lrratio},
                 {'params': self.model.res.parameters(), "lr": self.lr}]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(
            config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True, num_workers=config.num_workers)
        test_loader = data_loader.DataLoader(
            config.dataset, path, test_idx, config.patch_size, config.test_patch_num, batch_size=config.test_batch_size,  istrain=False, num_workers=config.num_workers)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0
        self.print('Epoch\tTrain_Loss Train_SRCC Test_SRCC Test_PLCC')
        for t in range(self.epochs):
            if t == 0:
                start_time = time.time()
            epoch_loss = []
            pred_scores = []
            gt_scores = []
            train_srcc = []

            self.model.train()

            # Isn't using tqdm
            if self.use_tqdm:
                train_data = tqdm(self.train_data, leave=False)
            else:
                train_data = self.train_data

            for imgs_cpu, label in train_data:
                gt_scores = gt_scores + label.tolist()
                self.solver.zero_grad()
                imgs_cuda = []
                for img in imgs_cpu:
                    imgs_cuda.append(img.to(self.device))

                vec, paras = self.model(imgs_cuda)

                model_target = hyper_model.TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False
                pred = model_target(vec).reshape(-1)

                loss = self.l1_loss(pred, label.to(self.device).detach())
                epoch_loss.append(loss.item())
                pred_scores = pred_scores + pred.cpu().tolist()

                loss.backward()
                self.solver.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            test_srcc, test_plcc = self.test(self.test_data)

            if t == 0:
                self.print(
                    f"epoch {t+1} 测试结束用时为：{(time.time()-start_time)/60} minute")

            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
            self.print('%d\t%4.3f\t%4.4f\t%4.4f\t%4.4f' %
                       (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))

            lr = self.lr / pow(10, (t // 1))
            # # if t > 4:
            # self.lrratio = 1

            paras = [{'params': self.wave_params, "lr": lr},
                     {'params': self.model.res.parameters(), "lr": lr}]
            self.solver = torch.optim.Adam(
                paras, weight_decay=self.weight_decay)

        self.print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        pred_scores = []
        test_srcc = []
        test_plcc = []
        gt_scores = []

        self.model.eval()
        with torch.no_grad():

            # Isn't using tqdm
            if self.use_tqdm:
                test_data = tqdm(data, leave=False)
            else:
                test_data = data

            for imgs_cpu, label in test_data:
                imgs_cuda = []
                for img in imgs_cpu:
                    imgs_cuda.append(img.to(self.device))
                vec, paras = self.model(imgs_cuda)

                model_target = hyper_model.TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False
                pred = model_target(vec).reshape(-1)

                gt_scores = gt_scores + label.tolist()
                pred_scores += pred.reshape(-1).cpu().tolist()

        gt_scores = np.mean(np.reshape(np.array(gt_scores),
                            (-1, self.test_patch_num)), axis=1)
        pred_scores = np.mean(np.reshape(
            np.array(pred_scores), (-1, self.test_patch_num)), axis=1)

        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        return test_srcc, test_plcc
