import torch

from tools.box import xywh2xyxy, bbox_iou


# Remove the overlapping boxes
# return: (x1, y1, x2, y2, object_conf, class_score)
def non_max_suppression(pred, conf_thres=0.5, nms_thres=0.4):
    pred[..., :4] = xywh2xyxy(pred[..., :4])
    result = [None for i in range(len(pred))]

    for id_img, image_pred in enumerate(pred):
        # Filter by conf
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # No objects
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by score DESC
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat(
            (image_pred[:, :5], class_confs.float(), class_preds.float()), 1
        )
        # Non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = (
                bbox_iou(detections[0, :4].unsqueeze(0),
                         detections[:, :4]) > nms_thres
            )
            # Check coincidence
            label_match = detections[0, -1] == detections[:, -1]
            # indexs with good, unique IoU
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # marge result
            detections[0, :4] = (weights * detections[invalid, :4]).sum(
                0
            ) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        # Set nms
        if keep_boxes:
            result[id_img] = torch.stack(keep_boxes)

    return result
