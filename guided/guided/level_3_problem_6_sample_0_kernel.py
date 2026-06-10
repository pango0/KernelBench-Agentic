import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class Conv2dCustom(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.save_for_backward(input, weight, bias, stride, padding, dilation, groups)
        output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias, stride, padding, dilation, groups = ctx.saved_tensors
        grad_input = None
        grad_weight = None
        grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = F.conv2d(grad_output, weight.flip(2).flip(3), None, stride, padding, dilation, groups)
        
        if ctx.needs_input_grad[1]:
            grad_weight = F.conv2d(input, grad_output.transpose(0, 1).flip(2).flip(3), None, stride, padding, dilation, groups)
            if bias is not None:
                grad_weight += bias.view(-1, 1, 1, 1)
        
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = grad_output.sum((0, 2, 3))
        
        return grad_input, grad_weight, grad_bias, None, None, None, None

class MaxPool2dCustom(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode
        output = F.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
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
        
        grad_input = F.max_unpool2d(grad_output, input.argmax(dim=1, keepdim=True), kernel_size, stride, padding, dilation, ceil_mode)
        
        return grad_input, None, None, None, None, None

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(ModelNew, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            Conv2dCustom.apply,
            nn.ReLU(inplace=True),
            Conv2dCustom.apply
        )
        
        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            Conv2dCustom.apply,
            nn.ReLU(inplace=True),
            Conv2dCustom.apply
        )
        
        # Max pooling branch
        self.branch_pool = nn.Sequential(
            MaxPool2dCustom.apply,
            Conv2dCustom.apply
        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)