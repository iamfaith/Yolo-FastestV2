import os
import cv2
# import time
import onnxruntime
# import torch
# import my_model.detector
# import utils.utils
import numpy as np
from time import time

from scipy.special import expit, softmax



def load_datafile(data_path):
    #需要配置的超参数
    cfg = {"model_name":None,
    
           "epochs": None,
           "steps": None,           
           "batch_size": None,
           "subdivisions":None,
           "learning_rate": None,

           "pre_weights": None,        
           "classes": None,
           "width": None,
           "height": None,           
           "anchor_num": None,
           "anchors": None,

           "val": None,           
           "train": None,
           "names":None
        }

    assert os.path.exists(data_path), "请指定正确配置.data文件路径"

    #指定配置项的类型
    list_type_key = ["anchors", "steps"]
    str_type_key = ["model_name", "val", "train", "names", "pre_weights"]
    int_type_key = ["epochs", "batch_size", "classes", "width",
                   "height", "anchor_num", "subdivisions"]
    float_type_key = ["learning_rate"]
    
    #加载配置文件
    with open(data_path, 'r') as f:
        for line in f.readlines():
            if line == '\n' or line[0] == "[":
                continue
            else:
                data = line.strip().split("=")
                #配置项类型转换
                if data[0] in cfg:
                    if data[0] in int_type_key:
                       cfg[data[0]] = int(data[1])
                    elif data[0] in str_type_key:
                        cfg[data[0]] = data[1]
                    elif data[0] in float_type_key:
                        cfg[data[0]] = float(data[1])
                    elif data[0] in list_type_key:
                        cfg[data[0]] = [float(x) for x in data[1].split(",")]
                    else:
                        print("配置文件有错误的配置项")
                else:
                    print("%s配置文件里有无效配置项:%s"%(data_path, data))
    return cfg


def make_grid(h, w, cfg):
    # hv: 行索引, wv: 列索引
    hv, wv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    grid = np.stack((wv, hv), axis=2)  # shape (h, w, 2)
    # repeat anchor_num 次
    grid = np.repeat(grid[:, :, np.newaxis, :], cfg["anchor_num"], axis=2)
    return grid.astype(np.float32)  # shape (h, w, anchor_num, 2)

def handel_preds(preds, cfg):
    # 加载 anchors
    anchors = np.array(cfg["anchors"], dtype=np.float32)
    anchors = anchors.reshape(len(preds) // 3, cfg["anchor_num"], 2)

    output_bboxes = []

    for i in range(len(preds) // 3):
        batch_bboxes = []
        reg_preds = preds[i * 3]
        obj_preds = preds[(i * 3) + 1]
        cls_preds = preds[(i * 3) + 2]

        for r, o, c in zip(reg_preds, obj_preds, cls_preds):
            # 转换维度 (H,W,C) -> (H,W,anchor_num,features)
            r = np.transpose(r, (1, 2, 0))
            r = r.reshape(r.shape[0], r.shape[1], cfg["anchor_num"], -1)

            o = np.transpose(o, (1, 2, 0))
            o = o.reshape(o.shape[0], o.shape[1], cfg["anchor_num"], -1)

            c = np.transpose(c, (1, 2, 0))
            c = c.reshape(c.shape[0], c.shape[1], 1, c.shape[2])
            c = np.repeat(c, cfg["anchor_num"], axis=2)

            # anchor_boxes: (H,W,anchor_num, 4+1+num_classes)
            anchor_boxes = np.zeros(
                (r.shape[0], r.shape[1], r.shape[2], r.shape[3] + c.shape[3] + 1),
                dtype=np.float32
            )

            # 计算 cx, cy
            grid = make_grid(r.shape[0], r.shape[1], cfg)
            stride = cfg["height"] / r.shape[0]
            anchor_boxes[:, :, :, :2] = ((expit(r[:, :, :, :2]) * 2.0 - 0.5) + grid) * stride

            # 计算 w, h
            anchors_cfg = anchors[i]
            anchor_boxes[:, :, :, 2:4] = (expit(r[:, :, :, 2:4]) * 2.0) ** 2 * anchors_cfg

            # 计算 obj 分数
            anchor_boxes[:, :, :, 4] = expit(o[:, :, :, 0])

            # 计算 cls 分数
            cls_scores = softmax(c, axis=3)
            anchor_boxes[:, :, :, 5:] = cls_scores

            batch_bboxes.append(anchor_boxes)

        # (N, H, W, anchor_num, box) -> (N, H*W*anchor_num, box)
        batch_bboxes = np.array(batch_bboxes, dtype=np.float32)
        batch_bboxes = batch_bboxes.reshape(batch_bboxes.shape[0], -1, batch_bboxes.shape[-1])

        output_bboxes.append(batch_bboxes)

    # merge along axis=1
    output = np.concatenate(output_bboxes, axis=1)
    return output


def xywh2xyxy(x):
    # x: [N,4] (cx, cy, w, h)
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

def bbox_iou(box1, box2):
    """
    box1: [N,4], box2: [M,4]
    return IoU matrix [N,M]
    """
    # Intersection
    inter_x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter_area = inter_w * inter_h

    # Union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - inter_area

    return inter_area / (union + 1e-16)

def nms(boxes, scores, iou_thres):
    """
    Pure numpy NMS
    boxes: [N,4]
    scores: [N]
    return keep indices
    """
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = bbox_iou(boxes[i:i+1], boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious <= iou_thres]
    return np.array(keep, dtype=np.int32)

def non_max_suppression(prediction, conf_thres=0.3, iou_thres=0.45, classes=None):
    """
    NumPy implementation of NMS
    prediction: [batch, num_boxes, 5+num_classes]
    Returns: list of detections per image, each [N,6] (x1,y1,x2,y2,conf,cls)
    """
    nc = prediction.shape[2] - 5
    max_wh = 4096
    max_det = 300
    max_nms = 30000
    time_limit = 1.0
    multi_label = nc > 1

    t = time()
    output = [np.zeros((0, 6))] * prediction.shape[0]

    for xi, x in enumerate(prediction):
        # Filter by confidence
        x = x[x[:, 4] > conf_thres]
        if not x.shape[0]:
            continue

        # Compute conf = obj_conf * cls_conf
        x[:, 5:] *= x[:, 4:5]

        # Convert boxes
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        j = np.argmax(x[:, 5:], axis=1)
        conf = x[np.arange(len(x)), 5 + j]
        mask = conf > conf_thres
        x = np.concatenate((box, conf[:, None], j[:, None].astype(np.float32)), axis=1)[mask]

        # Filter by class
        if classes is not None:
            x = x[np.isin(x[:, 5].astype(int), classes)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort()[::-1][:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        keep = nms(boxes, scores, iou_thres)
        if keep.shape[0] > max_det:
            keep = keep[:max_det]

        output[xi] = x[keep]

        if (time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return output


if __name__ == '__main__':

    cfg = load_datafile('/home/faith/Yolo-FastestV2/data/coco.data')
  
    # assert os.path.exists(opt.img), "请指定正确的测试图像路径"

    #模型加载
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # model = my_model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    # model.load_state_dict(torch.load(opt.weights, map_location=device))

    # #sets the module in eval node
    # model.eval()

    model_path = "yolo-fastestv2.onnx"
    session = onnxruntime.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            # Get model info
    output_names = [x.name for x in session.get_outputs()]
    input_names = [x.name for x in session.get_inputs()]
    

    from glob import glob

    times = []

    for img_path in glob('/home/faith/coco2017labels-person/coco/images/val/*.jpg'):
    # for img_path in glob('/home/faith/yolov5/data/images/*.jpg'):
        
        #数据预处理
        ori_img = cv2.imread(img_path)
        res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation = cv2.INTER_LINEAR) 
        img = res_img.reshape(1, cfg["height"], cfg["width"], 3)

        
        # image = resized_image.transpose(2, 0, 1)  # Convert from HWC -> CHW
        image = img.transpose(0,3, 1, 2)  # Convert from HWC -> CHW
        # image = image[::-1]  # Convert BGR to RGB
        image = np.ascontiguousarray(image)
        image = image.astype(np.float32) / 255.0  # Normalize the input
            
            
        # img = torch.from_numpy(img.transpose(0,3, 1, 2))
        # img = img.to(device).float() / 255.0

        start = time()
        #模型推理
        # start = time.perf_counter()
        # preds = model(img)
        preds = session.run(output_names, {input_names[0]: image})
        # end = time.perf_counter()
        # time = (end - start) * 1000.
        # print("forward time:%fms"%time)

        #特征图后处理
        # output = utils.utils.handel_preds(preds, cfg, "cpu")
        # output = output.detach().cpu().numpy()
        output = handel_preds(preds, cfg)
        output_boxes = non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)
        
        end = time()
        inference_time = end - start
        times.append(inference_time)
        print(f"Processing {img_path} Total Time: {inference_time * 1000:.2f} ms")
    # break


        # #加载label names
        # LABEL_NAMES = []
        # with open(cfg["names"], 'r') as f:
        #     for line in f.readlines():
        #         LABEL_NAMES.append(line.strip())
        
        # h, w, _ = ori_img.shape
        # scale_h, scale_w = h / cfg["height"], w / cfg["width"]

        # #绘制预测框
        # for box in output_boxes[0]:
        #     box = box.tolist()
        
        #     obj_score = box[4]
        #     category = LABEL_NAMES[int(box[5])]

        #     x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        #     x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

        #     cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        #     cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
        #     cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

        # cv2.imwrite("test_result_onnx.png", ori_img)
    
    

    if times:
        avg_time = np.mean(times)
        print(f"Average inference time: {avg_time * 1000:.2f} ms over {len(times)} images")
    

# Average inference time: 7.44 ms over 2693 images
# Average inference time: 7.74 ms over 2693 images
# Average inference time: 3.88 ms over 2693 images


#Average inference time: 4.38 ms over 2693 images