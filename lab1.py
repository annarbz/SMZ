import numpy as np

def convolution2D(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
   
    # Extract dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape

    # Apply padding
    padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)

    # Compute output dimensions
    output_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    output_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1

    # Initialize output tensor
    output_tensor = np.zeros((batch_size, out_channels, output_height, output_width), dtype=np.float32)

    # Perform convolution
    for b in range(batch_size):
        for o in range(out_channels):
            for i in range(in_channels):
                for h in range(0, input_height - kernel_height + 1, stride):
                    for w in range(0, input_width - kernel_width + 1, stride):
                        output_tensor[b, o, h // stride, w // stride] += np.sum(
                            padded_input[b, i, h:h + kernel_height, w:w + kernel_width] * weight[o, i])

    # Add bias if provided
    if bias is not None:
        output_tensor += bias.reshape(1, -1, 1, 1)

    return output_tensor

# import torch

# def convolution2D(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
#     # Преобразование входных данных в тензоры PyTorch
#     input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
#     weight = torch.tensor(weight, dtype=torch.float32)
#     if bias is not None:
#         bias = torch.tensor(bias, dtype=torch.float32)

#     # Применение свертки с использованием PyTorch
#     output_tensor = torch.nn.functional.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)

#     # Преобразование обратно в массив NumPy
#     output_tensor = output_tensor.numpy()

#     return output_tensor
