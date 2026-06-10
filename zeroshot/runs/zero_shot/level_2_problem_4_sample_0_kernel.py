import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class MishFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input * torch.tanh(torch.nn.functional.softplus(input))

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigmoid = torch.sigmoid(input)
        tanh_sp = torch.tanh(torch.nn.functional.softplus(input))
        grad_input = grad_output * (sigmoid + tanh_sp * (1 - sigmoid**2))
        return grad_input

mish = MishFunction.apply

class ConvMish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvMish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = mish(x)
        return x

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv1 = ConvMish(in_channels, out_channels, kernel_size)
        self.conv2 = ConvMish(out_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# Example usage:
batch_size   = 64  
in_channels  = 64  
out_channels = 128  
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]