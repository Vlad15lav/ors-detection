import torch
import torch.nn as nn
import torch.nn.functional as F

# Upsample layer
class Upsample(nn.Module): 
  def __init__(self, scale_factor, mode="nearest"):
      super(Upsample, self).__init__()
      self.scale_factor = scale_factor
      self.mode = mode

  def forward(self, x):
      x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
      return x

# For Residual module 
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
 
class PredictLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_size=512):
        super(PredictLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1 # weight obj
        self.noobj_scale = 100 # weight not obj
        self.metrics = {}
        self.img_size = img_size
        self.grid_size = 0 

    # Calculation bias
    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_size / self.grid_size

        self.grid_x = torch.arange(g).repeat(g, 1).\
            view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().\
            view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride)
            for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
 
    def forward(self, x, targets=None, img_size=None):
        # if cuda is enable
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
 
        self.img_size = img_size
        num_samples = x.size(0)
        grid_size = x.size(2)
 
        prediction = (x.view(num_samples, self.num_anchors, self.num_classes + 5,
            grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous())
 
        # Loss calculation
        x = torch.sigmoid(prediction[..., 0])  # X center
        y = torch.sigmoid(prediction[..., 1])  # Y center
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Class
 
        # Change grid size
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)
 
        # Add offset
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        # Predict
        output = torch.cat((pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
                ), -1, )
        
        if targets is None:
            return output, 0 # Predict
        else:
            # Train
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf\
            = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )
 
            # Losses
            obj_mask=obj_mask.bool()
            noobj_mask=noobj_mask.bool()
            
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj +\
                self.noobj_scale * loss_conf_noobj
            
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            return output, total_loss