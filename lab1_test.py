import torch
import torch.nn.functional as F
from lab1 import convolution2D

def test_custom_conv2d_1():
    # Генерация случайных входных данных и фильтров
    input_tensor = torch.randint(0, 6, (2, 3, 4, 4)).float()
    filters = torch.randint(0, 6, (2, 3, 2, 2)).float()

    # Выполнение свертки с использованием PyTorch
    conv_torch = F.conv2d(input_tensor, filters)

    # Выполнение свертки с использованием моей реализации
    conv_custom = convolution2D(input_tensor.numpy(), filters.numpy())

    # Преобразование массива numpy в тензор torch
    conv_custom = torch.tensor(conv_custom)

    # Сравнение результатов
    assert torch.allclose(conv_torch, conv_custom)

def test_custom_conv2d_2():
    # Генерация случайных входных данных и фильтров
    input_tensor = torch.randint(0, 6, (2, 3, 4, 4)).float()
    filters = torch.randint(0, 6, (2, 3, 2, 2)).float()
    
    # Выполнение свертки с использованием PyTorch с учетом смещения (bias)
    bias = torch.randint(0, 6, (2,)).float()
    conv_torch = F.conv2d(input_tensor, filters, bias=bias)

    # Выполнение свертки с использованием моей реализации с учетом смещения (bias)
    conv_custom = convolution2D(input_tensor.numpy(), filters.numpy(), bias=bias.numpy())

    # Преобразование массива numpy в тензор torch
    conv_custom = torch.tensor(conv_custom)

    # Сравнение результатов
    assert torch.allclose(conv_torch, conv_custom)

if __name__ == "__main__":
    test_custom_conv2d_1()
    test_custom_conv2d_2()
    print("Тесты пройдены!")
