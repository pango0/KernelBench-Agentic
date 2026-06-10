import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class CustomConv3dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = torch.nn.functional.conv3d(input, weight, bias, stride=1, padding=1, dilation=1, groups=1)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.conv_transpose3d(grad_output, weight, bias=None, stride=1, padding=1, output_padding=0, groups=1, dilation=1)
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.conv3d(input, grad_output, bias=None, stride=1, padding=1, dilation=1, groups=1)
        
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = torch.sum(grad_output, dim=(0, 2, 3, 4))
        
        return grad_input, grad_weight, grad_bias

class CustomActivationFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, activation_type):
        ctx.activation_type = activation_type
        if activation_type == 'relu':
            output = torch.relu(input)
        elif activation_type == 'leaky_relu':
            output = torch.nn.functional.leaky_relu(input, negative_slope=0.01)
        elif activation_type == 'gelu':
            output = torch.nn.functional.gelu(input)
        elif activation_type == 'sigmoid':
            output = torch.sigmoid(input)
        else:
            raise ValueError("Unsupported activation type")
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        if ctx.activation_type == 'relu':
            grad_input = grad_output * (input > 0).float()
        elif ctx.activation_type == 'leaky_relu':
            grad_input = grad_output * (input >= 0).float() + grad_output * (-0.01) * (input < 0).float()
        elif ctx.activation_type == 'gelu':
            # Approximate gradient of gelu
            cdf = 0.5 * (1 + torch.erf(input / torch.sqrt(2.0)))
            pdf = torch.exp(-0.5 * input ** 2) / torch.sqrt(2 * torch.pi)
            grad_input = grad_output * ((input * pdf) + (cdf * (1 - pdf)))
        elif ctx.activation_type == 'sigmoid':
            sigmoid = torch.sigmoid(input)
            grad_input = grad_output * sigmoid * (1 - sigmoid)
        else:
            raise ValueError("Unsupported activation type")
        return grad_input, None

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = CustomConv3dFunction.apply
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape[0]))

    def forward(self, x):
        x = self.conv(x, self.weight, self.bias)
        x = CustomActivationFunction.apply(x, 'relu')
        x = CustomActivationFunction.apply(x, 'leaky_relu')
        x = CustomActivationFunction.apply(x, 'gelu')
        x = CustomActivationFunction.apply(x, 'sigmoid')
        x = x + self.bias.view(1, -1, 1, 1, 1)
        return x