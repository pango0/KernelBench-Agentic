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
        grad_input = grad_output * (sigmoid + tanh_sp * (1 - sigmoid * tanh_sp))
        return grad_input

class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)

class Conv2dMish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dMish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mish = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.mish(x)
        return x

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv1 = Conv2dMish(in_channels, out_channels, kernel_size)
        self.conv2 = Conv2dMish(out_channels, out_channels, kernel_size)

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

model_new = ModelNew(in_channels, out_channels, kernel_size)
inputs = get_inputs()
output = model_new(inputs[0])
print(output.shape)