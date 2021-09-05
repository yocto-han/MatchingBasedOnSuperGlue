import argparse
import numpy as np
import pandas as pd
from torch.autograd import Variable
from dataloader import SparseDataset
from dataloaderfortestdata import SparseTestDataset
import torch.multiprocessing
from utils import confusion_matrix
from models.superglue import SuperGlue
from utils import viz
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')
# 这个策略将提供文件名称给shm_open去定义共享内存区域

# 以下为入模参数说明
parser = argparse.ArgumentParser(
    description='Data pair matching and pose evaluation with SuperGlue',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument(
    '--match_threshold', type=float, default=0.02,
    help='SuperGlue match threshold')

# 训练参数声明
parser.add_argument(
    '--learning_rate', type=int, default=0.002,
    help='Learning rate')
parser.add_argument(
    '--batch_size', type=int, default=1,
    help='batch_size')
parser.add_argument(
    '--train_path', type=str, default='data172/',
    help='Path to the directory of training data.')
parser.add_argument(
    '--epoch', type=int, default=1,
    help='Number of epoches')

# 测试参数声明
parser.add_argument(
    '--test_path', type=str, default='test/',
    help='Path to the directory of testing data.')

if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)

    # SuperPoint模块和SuperGlue模块参数
    config = {
        'superglue': {
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    # load training data and testdata
    train_set = SparseDataset(opt.train_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, batch_size=opt.batch_size,
                                               drop_last=True)
    # 调用SuperGlue模块，输入初始参数
    superglue = SuperGlue(config.get('superglue', {}))
    # # 确保训练过程在GPU上进行
    if torch.cuda.is_available():
        superglue.cuda()  # make sure it trains on GPU
    else:
        print("### CUDA not available ###")
    # 构建一个优化器optimizer，用Adam方法对于参数进行调优
    optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
    mean_loss = []
    matches = []
    results = []
    kpts0 = []
    kpts1 = []

    # start training
    # 开始进行训练
    max_acc = 0
    max_epoch = 0
    for epoch in range(1, opt.epoch + 1):
        epoch_loss = 0
        superglue.double().train()
        acc = []
        pre = []
        rec = []
        rec_dust = []
        P = 0
        N = 0
        TP = 0
        TN = 0
        for i, pred in enumerate(train_loader):
            for k in pred:
                # 此处控制k的格式是否正常
                if k != 'file_name':
                    if type(pred[k]) == torch.Tensor:
                        pred[k] = Variable(pred[k].cuda())
                    else:
                        pred[k] = Variable(torch.stack(pred[k]).cuda())

            # 将train_loader中的某一对数据放入SuperGlue模块
            data = superglue(pred)
            for k, v in pred.items():
                pred[k] = v[0]
            pred = {**pred, **data}

            # process loss
            Loss = pred['loss']
            acc.append(pred['acc'])
            pre.append(pred['pre'])
            rec.append(pred['rec'])
            rec_dust.append(pred['rec_dust'])

            epoch_loss += Loss.item()
            mean_loss.append(Loss)

            superglue.zero_grad()
            Loss.backward()
            optimizer.step()

            # acc_lst.append(pred['true'][0])
            if(epoch == 50):
                results.append(pred['results'])
                matches.append(pred['matches'])
                kpts0.append(pred['loc0'])
                kpts1.append(pred['loc1'])
                TP += pred['TP']
                TN += pred['TN']
                P += pred['P']
                N += pred['N']
        acc_epoch = np.mean(acc)
        pre_epoch = np.mean(pre)
        recall_epoch = np.mean(rec)
        recall_dust_epoch = np.mean(rec_dust)
        if acc_epoch > max_acc:
            max_acc = acc_epoch
            max_epoch = epoch
        epoch_loss /= len(train_loader)
        model_out_path = "parameter/model_epoch_{}.pth".format(epoch)
        torch.save(superglue.state_dict(), model_out_path)
        print("Epoch [{}/{}] done. Epoch Loss {}. Checkpoint saved to {}"
              .format(epoch, opt.epoch, epoch_loss, model_out_path))
        print("Epoch [{}/{}] done. Epoch Acc {:.4f}.".format(epoch, opt.epoch, acc_epoch))
        print("Epoch [{}/{}] done. Epoch Pre {:.4f}.".format(epoch, opt.epoch, pre_epoch))
        print("Epoch [{}/{}] done. Epoch Recall {:.4f}.".format(epoch, opt.epoch, recall_epoch))
        print("Epoch [{}/{}] done. Epoch Recall_global {:.4f}.".format(epoch, opt.epoch, recall_dust_epoch))
        if epoch == 50:
            viz(matches, results, kpts0, kpts1)
            confusion_matrix(P, N, TP, TN)
    print("Max_acc : {:.4f}, Epoch {}".format(max_acc, max_epoch))


    # 测试部分，将训练部分准确率最高的参数传入测试部分进行测试
    test_set = SparseTestDataset(opt.test_path)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=False, batch_size=opt.batch_size,
                                               drop_last=False)
    superglue1 = SuperGlue(config.get('superglue', {}))
    superglue1.double()
    superglue1.load_state_dict(torch.load('parameter/model_epoch_' + str(max_epoch) + ".pth"))

    for epoch in range(1, 2):
        with torch.no_grad():
            desc = []
            global_id = []
            for i, pred in enumerate(test_loader):
                for k in pred:
                    if k != 'file_name':
                        if type(pred[k]) == torch.Tensor:
                            pred[k] = Variable(pred[k])
                        else:
                            pred[k] = Variable(torch.stack(pred[k]))

                # 将test_loader中的某一对数据放入SuperGlue模块
                data = superglue1(pred)
                for k, v in pred.items():
                    pred[k] = v[0]
                pred = {**pred, **data}
                results = pred['results']
                id = pred['id'].numpy()
                for j in range(len(id)):
                    message = str(i + 1) + "#" + str(j) + "#" + str(id[j][0])
                    global_id.append(message)
                for k in range(len(pred['desc'][0].numpy())):
                    desc.append(pred['desc'][0].numpy()[k])
                # 查看测试结果
                print(pred['results'])

            # 导出深度描述符，利用DNSCAN聚类方法衡量描述符有效性
            desc = pd.DataFrame(desc, columns=["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8"])
            desc['global_id'] = global_id
            desc.to_csv("DescriptorAnalysis/desc_v5.csv")
            print(desc)