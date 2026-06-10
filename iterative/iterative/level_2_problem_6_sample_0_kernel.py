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
            grad_input = torch.nn.functional.conv_transpose3d(grad_output, weight, bias, stride=1, padding=1, output_padding=0, groups=1)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.conv3d(input, grad_output, None, stride=1, padding=1, dilation=1, groups=1)
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = torch.sum(grad_output, dim=(0, 2, 3, 4))
        return grad_input, grad_weight, grad_bias

class CustomMaxPool3dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode
        output = torch.nn.functional.max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        ceil_mode = ctx.ceil_mode
        grad_input = torch.zeros_like(input)
        indices = torch.argmax(input.view(input.size(0), input.size(1), -1).contiguous().view(-1), dim=-1)
        indices = indices.view(input.size(0), input.size(1), *input.size()[2:])
        grad_input.scatter_(dim=2, index=indices.unsqueeze(2), src=grad_output.contiguous())
        return grad_input, None, None, None, None, None

class CustomSoftmaxFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        exp_input = torch.exp(input - torch.max(input, dim=1, keepdim=True)[0])
        sum_exp = torch.sum(exp_input, dim=1, keepdim=True)
        output = exp_input / sum_exp
        ctx.save_for_backward(output)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = output * (grad_output - torch.sum(grad_output * output, dim=1, keepdim=True))
        return grad_input

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = CustomConv3dFunction.apply
        self.pool1 = CustomMaxPool3dFunction.apply
        self.pool2 = CustomMaxPool3dFunction.apply
        self.softmax = CustomSoftmaxFunction.apply

    def forward(self, x):
        x = self.conv(x, torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size, device=x.device))
        x = self.softmax(x)
        x = self.pool1(x, pool_kernel_size)
        x = self.pool2(x, pool_kernel_size)
        return x

# Example usage
model_new = ModelNew(in_channels, out_channels, kernel_size, pool_kernel_size)
inputs = get_inputs()
output = model_new(inputs[0])
print(output.shape)