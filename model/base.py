import torch.nn as nn
import torch

from model.layers import Upsample, EmptyLayer, PredictLayer

# Support functions for create cfg dict
def addConv(bn, filters, size, stride, pad, activation):
    return {'type': 'convolutional', 'activation': activation, 
    'batch_normalize': bn, 'filters': filters, 'pad': pad, 'size': size, 'stride': stride}

def addResidual(activation, from_value):
    return {'type': 'residual', 'activation': activation, 'from': from_value}

def addUpsample(stride):
    return {'type': 'upsample', 'stride': stride}

def addRoute(layers):
    return {'type': 'route', 'layers': layers}

def addYolo(anchors, classes, ign_t, jitter, mask, num, random, truth_t):
    return {'type': 'yolo', 'anchors': anchors, 'classes': classes,
    'ignore_thresh': ign_t, 'jitter': jitter, 'mask': mask, 'num': num,
    'random': random, 'truth_thresh': truth_t}

# Create cfg dict
def create_layers(yolo_anchors, num_classes):
    model_cfg = []

    # Add Darknet-53
    model_cfg.append(addConv(1, 32, 3, 1, 1, 'leaky'))
    model_cfg.append(addConv(1, 64, 3, 2, 1, 'leaky'))

    for i in range(1):
        model_cfg.append(addConv(1, 32, 1, 1, 1, 'leaky'))
        model_cfg.append(addConv(1, 64, 3, 1, 1, 'leaky'))
        model_cfg.append(addResidual('linear', -3))

    model_cfg.append(addConv(1, 128, 3, 2, 1, 'leaky'))

    for i in range(2):
        model_cfg.append(addConv(1, 64, 1, 1, 1, 'leaky'))
        model_cfg.append(addConv(1, 128, 3, 1, 1, 'leaky'))
        model_cfg.append(addResidual('linear', -3))

    model_cfg.append(addConv(1, 256, 3, 2, 1, 'leaky'))

    for i in range(8):
        model_cfg.append(addConv(1, 128, 1, 1, 1, 'leaky'))
        model_cfg.append(addConv(1, 256, 3, 1, 1, 'leaky'))
        model_cfg.append(addResidual('linear', -3))

    model_cfg.append(addConv(1, 512, 3, 2, 1, 'leaky'))

    for i in range(8):
        model_cfg.append(addConv(1, 256, 1, 1, 1, 'leaky'))
        model_cfg.append(addConv(1, 512, 3, 1, 1, 'leaky'))
        model_cfg.append(addResidual('linear', -3))

    model_cfg.append(addConv(1, 1024, 3, 2, 1, 'leaky'))

    for i in range(4):
        model_cfg.append(addConv(1, 512, 1, 1, 1, 'leaky'))
        model_cfg.append(addConv(1, 1024, 3, 1, 1, 'leaky'))
        model_cfg.append(addResidual('linear', -3))

    # Predict layers
    # The first output
    for i in range(3):
        model_cfg.append(addConv(1, 512, 1, 1, 1, 'leaky'))
        model_cfg.append(addConv(1, 1024, 3, 1, 1, 'leaky'))
    model_cfg.append(addConv(0, (num_classes + 5) * 3, 1, 1, 1, 'linear'))
    model_cfg.append(addYolo(yolo_anchors, num_classes, .7, .3, [6,7,8], 9, 1, 1))
    
    # Upsample block W*2xH*2xD
    model_cfg.append(addRoute([-4]))
    model_cfg.append(addConv(1, 256, 1, 1, 1, 'leaky'))
    model_cfg.append(addUpsample(2))
    model_cfg.append(addRoute([-1, 61]))

    # The second ouput
    for i in range(3):
        model_cfg.append(addConv(1, 256, 1, 1, 1, 'leaky'))
        model_cfg.append(addConv(1, 512, 3, 1, 1, 'leaky'))
    model_cfg.append(addConv(0, (num_classes + 5) * 3, 1, 1, 1, 'linear'))
    model_cfg.append(addYolo(yolo_anchors, num_classes, .7, .3, [3,4,5], 9, 1, 1))

    # Upsample block W*2xH*2xD
    model_cfg.append(addRoute([-4]))
    model_cfg.append(addConv(1, 128, 1, 1, 1, 'leaky'))
    model_cfg.append(addUpsample(2))
    model_cfg.append(addRoute([-1, 36]))

    # The third output
    for i in range(3):
        model_cfg.append(addConv(1, 128, 1, 1, 1, 'leaky'))
        model_cfg.append(addConv(1, 256, 3, 1, 1, 'leaky'))
    model_cfg.append(addConv(0, (num_classes + 5) * 3, 1, 1, 1, 'linear'))
    model_cfg.append(addYolo(yolo_anchors, num_classes, .7, .3, [0,1,2], 9, 1, 1))

    return model_cfg

# Create model by cfg dict
def create_modules(module_defs, img_size):
    output_filters = [3]
    module_list = nn.ModuleList()
    
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
 
        # Find type layer
        if module_def["type"] == "convolutional":
            bn = module_def["batch_normalize"]
            filters = module_def["filters"]
            kernel_size = module_def["size"]
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=module_def["stride"],
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}",
                    nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
  
        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=module_def["stride"], mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)
 
        elif module_def["type"] == "route": 
            layers = module_def["layers"]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())
 
        elif module_def["type"] == "residual":
            filters = output_filters[1:][module_def["from"]]
            modules.add_module(f"residual_{module_i}", EmptyLayer())
 
        elif module_def["type"] == "yolo":
            anchor_idxs = module_def["mask"]
            # Create anchor boxes
            anchors = module_def["anchors"]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = module_def["classes"]
            # Create predict layer
            yolo_layer = PredictLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        
        module_list.append(modules)
        output_filters.append(filters)
 
    return module_list
