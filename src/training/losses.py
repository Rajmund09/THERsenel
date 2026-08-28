"""
Loss Functions for Object Detection & Multimodal Fusion (src/training/losses.py)
Implements CIoU (Complete Intersection over Union) box loss and Binary Cross-Entropy classification loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_ciou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes Complete IoU (CIoU) loss between predicted and ground-truth boxes in format [x_center, y_center, w, h].
    """
    px, py, pw, ph = pred_boxes.unbind(-1)
    tx, ty, tw, th = target_boxes.unbind(-1)

    px1, py1 = px - pw / 2.0, py - ph / 2.0
    px2, py2 = px + pw / 2.0, py + ph / 2.0

    tx1, ty1 = tx - tw / 2.0, ty - th / 2.0
    tx2, ty2 = tx + tw / 2.0, ty + th / 2.0

    # Intersection area
    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union area
    p_area = pw * ph
    t_area = tw * th
    union_area = p_area + t_area - inter_area + eps

    iou = inter_area / union_area

    # Enclosing box
    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch = torch.max(py2, ty2) - torch.min(py1, ty1)
    c2 = cw ** 2 + ch ** 2 + eps

    # Center distance
    rho2 = (px - tx) ** 2 + (py - ty) ** 2

    # Aspect ratio term
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2 + alpha * v)
    return 1.0 - ciou


class MultimodalDetectionLoss(nn.Module):
    """
    Combined Loss for object classification (BCE/Focal) and bounding box regression (CIoU).
    """

    def __init__(self, cls_weight: float = 1.0, box_weight: float = 2.5):
        super().__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_cls: torch.Tensor, pred_boxes: torch.Tensor, target_cls: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        loss_cls = self.bce_loss(pred_cls, target_cls)
        loss_box = bbox_ciou(pred_boxes, target_boxes).mean()
        return self.cls_weight * loss_cls + self.box_weight * loss_box
