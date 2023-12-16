import torch
import torch.nn.functional as F

class Convolution3DModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation_fn=None):
        super(Convolution3DModule, self).__init__()
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.conv3d(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x

def Convolution3D(input, weight, bias=None, stride=1, padding=0, activation_fn=None):
    # Проверка размерностей входных данных
    if input.dim() != 5 or weight.dim() != 5:
        raise ValueError("Input and weight must be 5-dimensional")

    # Изменение порядка размерностей для Layout NHWDC
    input = input.permute(0, 4, 1, 2, 3).contiguous()
    weight = weight.permute(4, 3, 0, 1, 2).contiguous()

    # Применение операции свертки 3D с использованием PyTorch
    output = F.conv3d(input, weight, bias, stride=stride, padding=padding)

    # Изменение порядка размерностей обратно
    output = output.permute(0, 2, 3, 4, 1).contiguous()

    # Применение функции активации, если она предоставлена
    if activation_fn is not None:
        output = activation_fn(output)

    return output
