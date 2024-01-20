import torch
import torch.nn as nn

from utils.dataset import get_dataset

def get_some_graph_data(dname, device):
    inputs = get_dataset(dname)
    inputs.to(device)
    return inputs


def get_layer_metric_array(net, metric, mode): 
    metric_array = []

    for layer in net.modules():
        if mode=='channel' and hasattr(layer,'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))
    
    return metric_array

def reshape_elements(elements, shapes, device):
    def broadcast_val(elements, shapes):
        ret_grads = []
        for e,sh in zip(elements, shapes):
            ret_grads.append(torch.stack([torch.Tensor(sh).fill_(v) for v in e], dim=0).to(device))
        return ret_grads
    if type(elements[0]) == list:
        outer = []
        for e,sh in zip(elements, shapes):
            outer.append(broadcast_val(e,sh))
        return outer
    else:
        return broadcast_val(elements, shapes)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)