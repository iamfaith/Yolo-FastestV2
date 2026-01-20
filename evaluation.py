import os
import torch
import argparse
from tqdm import tqdm


# from torchsummary import summary

from utils.utils_new import load_datafile, evaluation
import utils.datasets
import my_model.detector

# %%
if __name__ == '__main__':
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/faith/Yolo-FastestV2/data/coco.data',
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='/home/faith/Yolo-FastestV2/modelzoo/coco2017-0.241078ap-model.pth',
                        help='The path of the model')
    opt = parser.parse_args()
    cfg = load_datafile(opt.data)

    assert os.path.exists(opt.weights), "请指定正确的模型路径"

    #打印消息
    print("评估配置:")
    print("model_name:%s"%cfg["model_name"])
    print("width:%d height:%d"%(cfg["width"], cfg["height"]))
    print("val:%s"%(cfg["val"]))
    print("model_path:%s"%(opt.weights))
    
    #加载数据
    val_dataset = utils.datasets.TensorDataset(cfg["val"], cfg["width"], cfg["height"], imgaug = False)

    # # for testing
    # from torch.utils.data import Subset 
    # # 假设 val_dataset 已经构建好了 
    # small_indices = list(range(128)) # 只取前 50 个样本 
    # val_dataset = Subset(val_dataset, small_indices)


    batch_size = int(cfg["batch_size"] / cfg["subdivisions"])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 collate_fn=utils.datasets.collate_fn,
                                                 num_workers=nw,
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 persistent_workers=True
                                                 )
    
    #指定后端设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #初始化模型
    model = my_model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    #sets the module in eval node
    model.eval()
    
    # %%
    
    #打印模型结构
    # summary(model, input_size=(3, cfg["height"], cfg["width"]))
    
    #模型评估
    # print("computer mAP...")
    # _, _, AP, _ = utils.utils.evaluation(val_dataloader, cfg, model, device)
    # print("computer PR...")
    # precision, recall, _, f1 = utils.utils.evaluation(val_dataloader, cfg, model, device, 0.3)

    # Precision:0.436081 Recall:0.463394 AP:0.393459 F1:0.449323
    # precision, recall, AP, f1 = utils.utils.evaluation(val_dataloader, cfg, model, device, 0.15)

    
    # Precision:0.657613 Recall:0.381924 AP:0.348270 F1:0.483212
    # precision, recall, AP, f1 = utils.utils.evaluation(val_dataloader, cfg, model, device, 0.3)

    # Precision:0.093609 Recall:0.595435 AP:0.426000 F1:0.161784
    # precision, recall, AP, f1 = utils.utils.evaluation(val_dataloader, cfg, model, device)
    # print("Precision:%f Recall:%f AP:%f F1:%f"%(precision, recall, AP, f1))


# {'precision_last': 0.09312774430715817, 'recall_last': 0.5923726454486407, 'precision_mean': 0.2705574852054696, 'recall_mean': 0.5154541287827205, 'mAP@0.5': 0.42536753588186527, 'mAP@0.5:0.95': 0.19953325127166321, 'f1_last': 0.16095199677289226, 'f1_mean': 0.3548547383787754}
    # print(evaluation(val_dataloader, cfg, model, device))
    
    
    # {'precision_last': 0.4358190709046455, 'recall_last': 0.4631158949614921, 'precision_mean': 0.7167358475183917, 'recall_mean': 0.32294423924691046, 'mAP@0.5': 0.3932841228707663, 'mAP@0.5:0.95': 0.19121608857340638, 'f1_last': 0.4490530388231589, 'f1_mean': 0.44526333814464564}
    # print(evaluation(val_dataloader, cfg, model, device, 0.15))
    
    
    # {'precision_last': 0.5213475650433622, 'recall_last': 0.4350932541523615, 'precision_mean': 0.7823329723365015, 'recall_mean': 0.288054923628217, 'mAP@0.5': 0.37979629456467345, 'mAP@0.5:0.95': 0.18750575727274715, 'f1_last': 0.4743310909918567, 'f1_mean': 0.42107139934559756}
    # print(evaluation(val_dataloader, cfg, model, device, 0.2))

    
    # {'precision_last': 0.5977684038644714, 'recall_last': 0.4076273545513594, 'precision_mean': 0.8324732189355157, 'recall_mean': 0.2579809516325173, 'mAP@0.5': 0.36438094595649323, 'mAP@0.5:0.95': 0.183204386761334, 'f1_last': 0.4847180845194748, 'f1_mean': 0.39389501920598197}
    # print(evaluation(val_dataloader, cfg, model, device, 0.25))
    
    
    # {'precision_last': 0.657453267295095, 'recall_last': 0.3818316785747425, 'precision_mean': 0.8683720277719409, 'recall_mean': 0.23412763119484115, 'mAP@0.5': 0.3482085631085026, 'mAP@0.5:0.95': 0.1781068895871581, 'f1_last': 0.4830946231509744, 'f1_mean': 0.368816233555372}
    print(evaluation(val_dataloader, cfg, model, device, 0.3))

