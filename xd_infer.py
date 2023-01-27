import torch
import numpy as np
from dataset_loader import XDVideo
from options import parse_args
import pdb
from config import Config
import utils
import os
from model import WSAD
from tqdm import tqdm
from dataset_loader import data
from sklearn.metrics import roc_curve,auc,precision_recall_curve
def valid(net, config, test_loader, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))
            
        pre_dict = {}
        gt_dict = {}
        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/xd_gt.npy")
        frame_predict = None
        cls_label = []
        cls_pre = []
        count = 0
        for i in tqdm(range(len(test_loader.dataset)//5)):

            _data, _label = next(load_iter)
            
            _data = _data.cuda()
            _label = _label.cuda()

            cls_label.append(int(_label[0]))
            res = net(_data)   
            a_predict = res["frame"].cpu().numpy().mean(0)   
            cls_pre.append(1 if a_predict.max()>0.5 else 0)          
            fpre_ = np.repeat(a_predict,16)
            pl = len(fpre_)
            pre_dict[i] = fpre_
            gt_dict[i] = frame_gt[count: count+pl]
            count = count + pl
            if frame_predict is None:         
                frame_predict = fpre_
            else:
                frame_predict = np.concatenate([frame_predict, fpre_])   
        np.save('frame_label/xd_frame_pre.npy', frame_predict)
        np.save('frame_label/xd_pre_dict.npy', pre_dict)
        np.save('frame_label/xd_gt_dict.npy', gt_dict)
        fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
        print("auc:{}".format(auc_score))
        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict,)
        ap_score = auc(recall, precision)

        print("accuracy:{}".format(accuracy))
        print("ap_score:{}".format(ap_score))
         
if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()
    config = Config(args)
    worker_init_fn = None
    config.len_feature = 1024
    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)
    net = WSAD(config.len_feature, flag = "Test", a_nums = 60, n_nums = 60)
    net = net.cuda()
    test_loader = data.DataLoader(
        XDVideo(root_dir = config.root_dir, mode = 'Test', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature),
            batch_size = 5,
            shuffle = False, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn)
    valid(net, config, test_loader, model_file = os.path.join(args.model_path, "xd_trans_2022.pkl"))