import numpy as np

import math
import random

import torch
from torch.autograd import Variable


'''Functions taken from the torch.nn.init module of the latest version of Pytorch.'''



def xavier_uniform(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method described in "Understanding the
        difficulty of training deep feedforward neural networks" - Glorot, X. and Bengio, Y., using a uniform
        distribution. The resulting tensor will have values sampled from U(-a, a) where a = gain * sqrt(2/(fan_in +
        fan_out)) * sqrt(3)
        Args:
        tensor: a n-dimension torch.Tensor
        gain: an optional scaling factor to be applied
        Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.xavier_uniform(w, gain=math.sqrt(2.0))
        """
    if isinstance(tensor, Variable):
        xavier_uniform(tensor.data, gain=gain)
        return tensor
    
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return tensor.uniform_(-a, a)


def xavier_normal(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method described in "Understanding the
        difficulty of training deep feedforward neural networks" - Glorot, X. and Bengio, Y., using a normal
        distribution. The resulting tensor will have values sampled from normal distribution with mean=0 and std = gain *
        sqrt(2/(fan_in + fan_out))
        Args:
        tensor: a n-dimension torch.Tensor
        gain: an optional scaling factor to be applied
        Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.xavier_normal(w)
        """
    if isinstance(tensor, Variable):
        xavier_normal(tensor.data, gain=gain)
        return tensor
    
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return tensor.normal_(0, std)



def kaiming_uniform(tensor, a=0, mode='fan_in'):
    """Fills the input Tensor or Variable with values according to the method described in "Delving deep into
        rectifiers: Surpassing human-level performance on ImageNet classification" - He, K. et al using a uniform
        distribution. The resulting tensor will have values sampled from U(-bound, bound) where bound = sqrt(2/((1 + a^2)
        * fan_in)) * sqrt(3)
        Args:
        tensor: a n-dimension torch.Tensor
        a: the coefficient of the slope of the rectifier used after this layer (0 for ReLU by default)
        mode: either 'fan_in' (default) or 'fan_out'. Choosing `fan_in` preserves the magnitude of the variance of the
        weights in the forward pass. Choosing `fan_out` preserves the magnitudes in the backwards pass.
        Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.kaiming_uniform(w, mode='fan_in')
        """
    if isinstance(tensor, Variable):
        kaiming_uniform(tensor.data, a=a, mode=mode)
        return tensor
    
    fan = _calculate_correct_fan(tensor, mode)
    std = math.sqrt(2.0 / ((1 + a ** 2) * fan))
    bound = math.sqrt(3.0) * std
    return tensor.uniform_(-bound, bound)


def kaiming_normal(tensor, a=0, mode='fan_in'):
    """Fills the input Tensor or Variable with values according to the method described in "Delving deep into
        rectifiers: Surpassing human-level performance on ImageNet classification" - He, K. et al using a normal
        distribution. The resulting tensor will have values sampled from normal distribution with mean=0 and std = sqrt(
        2/((1 + a^2) * fan_in))
        Args:
        tensor: a n-dimension torch.Tensor
        a: the coefficient of the slope of the rectifier used after this layer (0 for ReLU by default)
        mode: either 'fan_in' (default) or 'fan_out'. Choosing `fan_in` preserves the magnitude of the variance of the
        weights in the forward pass. Choosing `fan_out` preserves the magnitudes in the backwards pass.
        Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.kaiming_normal(w, mode='fan_out')
        """
    if isinstance(tensor, Variable):
        kaiming_normal(tensor.data, a=a, mode=mode)
        return tensor
    
    fan = _calculate_correct_fan(tensor, mode)
    std = math.sqrt(2.0 / ((1 + a ** 2) * fan))
    return tensor.normal_(0, std)




def _calculate_fan_in_and_fan_out(tensor):
    if tensor.ndimension() < 2:
        raise ValueError("fan in and fan out can not be computed for tensor of size ", tensor.size())
    
    if tensor.ndimension() == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("mode {} not supported, please use one of {}".format(mode, valid_modes))
    
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        return fan_in
    else:
        return fan_out
