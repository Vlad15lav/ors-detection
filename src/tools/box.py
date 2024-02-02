import numpy as np
import torch


def rescale_boxes(boxes, current_dim, original_shape):
    """
    Scale boxes to fit a different image size
    """
    orig_h, orig_w = original_shape

    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # wh without pad
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # rescale
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor)\
        else np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor)\
        else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


# Metric IoU with two boxes
def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


# Metric IoU with array coords
def bbox_iou(box_a, box_b, x1y1x2y2=True):
    if not x1y1x2y2:
        # xywh2xyxy
        b1_x1, b1_x2 = box_a[:, 0] - box_a[:, 2] / 2, \
            box_a[:, 0] + box_a[:, 2] / 2
        b1_y1, b1_y2 = box_a[:, 1] - box_a[:, 3] / 2, \
            box_a[:, 1] + box_a[:, 3] / 2
        b2_x1, b2_x2 = box_b[:, 0] - box_b[:, 2] / 2, \
            box_b[:, 0] + box_b[:, 2] / 2
        b2_y1, b2_y2 = box_b[:, 1] - box_b[:, 3] / 2, \
            box_b[:, 1] + box_b[:, 3] / 2
    else:
        # xyxy
        b1_x1, b1_y1, b1_x2, b1_y2 = box_a[:, 0], box_a[:, 1], \
            box_a[:, 2], box_a[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box_b[:, 0], box_b[:, 1], \
            box_b[:, 2], box_b[:, 3]

    # Intersec coord
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersect Area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) *\
        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou
