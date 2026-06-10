import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class Conv2dCustom(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = torch.nn.functional.conv2d(input, weight, bias, stride=1, padding=0, dilation=1, groups=1)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = None
        grad_weight = None
        grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.conv2d(grad_output, weight.flip(dims=[2, 3]), None, stride=1, padding=weight.size(2)//2, dilation=1, groups=1)
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.conv2d(input, grad_output.flip(dims=[2, 3]), None, stride=1, padding=0, dilation=1, groups=input.size(1))
        
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = torch.sum(grad_output, dim=(0, 2, 3))
        
        return grad_input, grad_weight, grad_bias

class InceptionModuleCustom(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(InceptionModuleCustom, self).__init__()
        
        self.branch1x1 = Conv2dCustom.apply
        self.branch3x3 = nn.Sequential(
            Conv2dCustom.apply,
            Conv2dCustom.apply
        )
        self.branch5x5 = nn.Sequential(
            Conv2dCustom.apply,
            Conv2dCustom.apply
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv2dCustom.apply
        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x, self.branch1x1.weight, self.branch1x1.bias)
        branch3x3 = self.branch3x3(x, self.branch3x3[0].weight, self.branch3x3[0].bias)
        branch3x3 = self.branch3x3(branch3x3, self.branch3x3[1].weight, self.branch3x3[1].bias)
        branch5x5 = self.branch5x5(x, self.branch5x5[0].weight, self.branch5x5[0].bias)
        branch5x5 = self.branch5x5(branch5x5, self.branch5x5[1].weight, self.branch5x5[1].bias)
        branch_pool = self.branch_pool(x, self.branch_pool[1].weight, self.branch_pool[1].bias)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        self.conv1 = Conv2dCustom.apply
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = Conv2dCustom.apply
        self.conv3 = Conv2dCustom.apply
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception3a = InceptionModuleCustom(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModuleCustom(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception4a = InceptionModuleCustom(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModuleCustom(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModuleCustom(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModuleCustom(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModuleCustom(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception5a = InceptionModuleCustom(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModuleCustom(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(F.relu(self.conv3(x)))
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x