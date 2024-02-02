import numpy as np
import torch
import torch.nn as nn

from model.base import create_layers, create_modules


# Main class YOLOv3
class YoloV3(nn.Module):
    def __init__(self, num_classes, anchors, img_size=512):
        super(YoloV3, self).__init__()
        self.module_defs = create_layers(anchors, num_classes)
        self.module_list = create_modules(self.module_defs, img_size)
        self.yolo_layers = [
            layer[0] for layer in self.module_list
            if hasattr(layer[0], "metrics")
        ]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_size = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []

        # Forward layers
        for i, (module_def, module) in enumerate(
            zip(self.module_defs, self.module_list)
        ):
            if module_def["type"] in ["convolutional", "upsample"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat(
                    [layer_outputs[int(layer_i)]
                     for layer_i in module_def["layers"]], 1
                )
            elif module_def["type"] == "residual":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_size)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = torch.cat(yolo_outputs, 1).detach().cpu()

        return yolo_outputs if targets is None else (loss, yolo_outputs)

    # Load weights from darknet
    def load_darknet_weights(self, weights_path):
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # header name
            self.header_info = header  # header info block
            self.seen = header[3]  # Number of samples in train time
            weights = np.fromfile(f, dtype=np.float32)  # Weights

        ptr = 0
        for i, (module_def, module) in enumerate(
            zip(self.module_defs, self.module_list)
        ):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of weight bias
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.bias
                    )
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.weight
                    )
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.running_mean
                    )
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.running_var
                    )
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]) \
                        .view_as(
                        conv_layer.bias
                        )
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b

                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(
                    conv_layer.weight
                )
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
