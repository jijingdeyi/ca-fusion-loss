import torch
import torch.nn as nn
import torch.nn.init as init
import functools
from typing import cast


def init_weights(
    net: nn.Module, 
    init_type: str = 'xavier_uniform', 
    init_bn_type: str = 'uniform', 
    gain: float = 1.0
) -> None:
    """
    Initialize network weights with various initialization methods.
    
    Based on: Kai Zhang, https://github.com/cszn/KAIR
    
    Args:
        net: PyTorch model to initialize
        init_type: Initialization method for Conv/Linear layers
            - 'default' or 'none': skip initialization
            - 'normal': normal distribution
            - 'uniform': uniform distribution
            - 'xavier_normal': Xavier normal initialization
            - 'xavier_uniform': Xavier uniform initialization (default)
            - 'kaiming_normal': Kaiming normal initialization
            - 'kaiming_uniform': Kaiming uniform initialization
            - 'orthogonal': orthogonal initialization
        init_bn_type: Initialization method for BatchNorm layers
            - 'uniform': uniform distribution (default)
            - 'constant': constant initialization
        gain: Scaling factor for initialization (default: 1.0)
    
    Example:
        >>> model = MyModel()
        >>> init_weights(model, init_type='kaiming_normal', gain=0.2)
    """

    def init_fn(m: nn.Module, init_type: str = 'xavier_uniform', init_bn_type: str = 'uniform', gain: float = 1.0) -> None:
        """Initialize a single module."""
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            if not hasattr(m, 'weight') or m.weight is None:
                return
            
            weight = cast(torch.Tensor, m.weight.data)
            bias = cast(torch.Tensor, m.bias.data) if hasattr(m, 'bias') and m.bias is not None else None

            if init_type == 'normal':
                init.normal_(weight, 0, 0.1)
                weight.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(weight, -0.2, 0.2)
                weight.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(weight, gain=gain)
                weight.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(weight, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(weight, a=0, mode='fan_in', nonlinearity='relu')
                weight.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(weight, a=0, mode='fan_in', nonlinearity='relu')
                weight.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(weight, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if bias is not None:
                bias.zero_()

        elif classname.find('BatchNorm2d') != -1:
            if not hasattr(m, 'affine') or not m.affine:
                return
            
            weight = cast(torch.Tensor, m.weight.data)
            bias = cast(torch.Tensor, m.bias.data)

            if init_bn_type == 'uniform':  # preferred
                init.uniform_(weight, 0.1, 1.0)
                init.constant_(bias, 0.0)
            elif init_bn_type == 'constant':
                init.constant_(weight, 1.0)
                init.constant_(bias, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network defination!')