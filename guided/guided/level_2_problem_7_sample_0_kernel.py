import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class Conv3dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = torch.nn.functional.conv3d(input, weight, bias=bias, padding=1)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.conv3d(grad_output, weight.flip(2).flip(3).flip(4), padding=1)
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.conv3d(input.transpose(1, 0), grad_output, padding=1).transpose(1, 0)
        
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = torch.sum(grad_output, dim=(0, 2, 3, 4))

        return grad_input, grad_weight, grad_bias

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = Conv3dFunction.apply
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv(x, self.weight)
        x = torch.relu(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        x = torch.nn.functional.gelu(x)
        x = torch.sigmoid(x)
        x = x + self.bias
        return x

# Note: The weight parameter needs to be defined before using the model
model_new = ModelNew(in_channels, out_channels, kernel_size, bias_shape)
model_new.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))