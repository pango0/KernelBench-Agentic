import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class CustomConv3dFunction(Function):
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

class CustomActivationFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.nn.functional.relu(input)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0
        return grad_input

class CustomLeakyReLUFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, negative_slope=0.01):
        ctx.save_for_backward(input)
        output = torch.nn.functional.leaky_relu(input, negative_slope=negative_slope)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        neg_mask = input < 0
        pos_mask = ~neg_mask
        grad_input[neg_mask] *= negative_slope
        return grad_input, None

class CustomGELUFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.nn.functional.gelu(input)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        g = 0.5 * (1 + torch.erf(input / torch.sqrt(torch.tensor(2.0))))
        dg = 0.5 * torch.exp(-input ** 2) / torch.sqrt(torch.tensor(2.0))
        grad_input = grad_output * (g + input * dg)
        return grad_input

class CustomSigmoidFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.sigmoid(input)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        s = torch.sigmoid(input)
        ds = s * (1 - s)
        grad_input = grad_output * ds
        return grad_input

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = CustomConv3dFunction.apply
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x, self.weight)
        x = CustomActivationFunction.apply(x)
        x = CustomLeakyReLUFunction.apply(x, negative_slope=0.01)
        x = CustomGELUFunction.apply(x)
        x = CustomSigmoidFunction.apply(x)
        x = x + self.bias
        return x