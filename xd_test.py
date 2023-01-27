import torch
from options import *
from config import *
from model import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

def test(net, config, wind, test_loader, test_info, step, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/xd_gt.npy")
        frame_predict = None
        cls_label = []
        cls_pre = []
        for i in range(len(test_loader.dataset)//5):

            _data, _label = next(load_iter)
            
            _data = _data.cuda()
            _label = _label.cuda()
            cls_label.append(int(_label[0]))
            res = net(_data)   
        
            a_predict = res["frame"].cpu().numpy().mean(0)   
            cls_pre.append(1 if a_predict.max()>0.5 else 0)          
            fpre_ = np.repeat(a_predict, 16)
            if frame_predict is None:         
                frame_predict = fpre_
            else:
                frame_predict = np.concatenate([frame_predict, fpre_])   

        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
      
        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))
       
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict,)
        ap_score = auc(recall, precision)
      
        wind.plot_lines('roc_auc', auc_score)
        wind.plot_lines('accuracy', accuracy)
        wind.plot_lines('pr_auc', ap_score)
        wind.lines('scores', frame_predict)
        wind.lines('roc_curve', tpr, fpr)
        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)
        