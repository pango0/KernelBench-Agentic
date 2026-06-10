import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class CustomMatMulReLUFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = torch.matmul(input, weight)
        return F.relu(output)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, weight.t())
        grad_weight = torch.matmul(input.t(), grad_output)
        return grad_input, grad_weight

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, input):
        return CustomMatMulReLUFunction.apply(input, self.weight) + self.bias

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(ModelNew, self).__init__()
        
        layers = []
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            layers.append(CustomLinear(current_input_size, hidden_size))
            current_input_size = hidden_size
        
        layers.append(CustomLinear(current_input_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)