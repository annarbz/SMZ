import numpy as np
import torch
import torch.nn as nn

def conv3d_nhwdc(input_volume, kernel, stride=1, padding=0):
    """
    3D-свертка с распределением 'NHWDC', реализованная с использованием NumPy.

    Параметры:
    - input_volume: Входной 5D тензор с формой (batch_size, depth, height, width, channels)
    - kernel: 5D тензор ядра с формой (out_channels, kernel_depth, kernel_height, kernel_width, in_channels)
    - stride: Шаг операции свертки
    - padding: Заполнение для операции свертки

    Возвращает:
    - output_volume: Выходной 5D тензор после свертки
    """
    batch_size, in_depth, in_height, in_width, in_channels = input_volume.shape
    out_channels, kernel_depth, kernel_height, kernel_width, _ = kernel.shape

    # Вычисляем размеры выходного тензора
    out_depth = (in_depth - kernel_depth + 2 * padding) // stride + 1
    out_height = (in_height - kernel_height + 2 * padding) // stride + 1
    out_width = (in_width - kernel_width + 2 * padding) // stride + 1

    # Добавляем к входному тензору паддинг
    padded_input = np.pad(input_volume, ((0, 0), (padding, padding), (padding, padding), (padding, padding), (0, 0)), mode='constant')

    # Инициализируем выходной тензор
    output_volume = np.zeros((batch_size, out_depth, out_height, out_width, out_channels))

    # Выполняем 3D-свертку
    for d in range(0, in_depth, stride):
        for h in range(0, in_height, stride):
            for w in range(0, in_width, stride):
                input_patch = padded_input[:, d:d+kernel_depth, h:h+kernel_height, w:w+kernel_width, :]
                output_patch = np.sum(input_patch * kernel, axis=(1, 2, 3, 4), keepdims=True)
                output_volume[:, d//stride, h//stride, w//stride, :] = output_patch.squeeze()

    return output_volume


def compare_results(input_tensor_np, kernel_np):
    # Применяем 3D-свертку
    output_tensor_np = conv3d_nhwdc(input_tensor_np, kernel_np, stride=1, padding=1)

    # Применяем 3D-свертку PyTorch
    conv3d_torch = nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    output_tensor_torch = conv3d_torch(torch.tensor(input_tensor_np, dtype=torch.float32).permute(0, 4, 1, 2, 3))

    # Выводим размеры выходных тензоров
    print("Размер выходного тензора (NumPy):", output_tensor_np.shape)
    print("Размер выходного тензора (PyTorch):", output_tensor_torch.permute(0, 2, 3, 4, 1).shape)

    # Сравниваем значения
    print("Тензоры идентичны:", np.allclose(output_tensor_np, output_tensor_torch.detach().permute(0, 2, 3, 4, 1).numpy()))


# Тест
input_tensor_np = np.random.randn(32, 10, 128, 128, 3).astype(np.float32)
kernel_np = np.random.randn(32, 3, 3, 3, 3).astype(np.float32)
compare_results(input_tensor_np, kernel_np)
print("\n")
