import unittest
import numpy as np
from lab1 import convolution2D  

class TestConvolution2D(unittest.TestCase):
    def test_convolution(self):
        # Тестирование базовой свертки без смещения (bias)
        input_tensor = np.random.rand(2, 3, 4, 4)  # Размеры: Batch size: 2, In channels: 3, Height: 4, Width: 4
        weight = np.random.rand(4, 3, 3, 3)  # Размеры: Out channels: 4, In channels/groups: 3, Kernel size: 3x3

        output_tensor = convolution2D(input_tensor, weight)

        # Утверждения для проверки корректности output_tensor
        self.assertEqual(output_tensor.shape, (2, 4, 2, 2))  # Пример размеров выходного тензора
        

    def test_convolution_with_bias(self):
        # Тестирование свертки с использованием смещения (bias)
        input_tensor = np.random.rand(2, 3, 4, 4)
        weight = np.random.rand(4, 3, 3, 3)
        bias = np.random.rand(4)

        output_tensor = convolution2D(input_tensor, weight, bias)

        # Утверждения для проверки корректности output_tensor
        self.assertEqual(output_tensor.shape, (2, 4, 2, 2))  # Пример размеров выходного тензора
        

    def test_convolution_with_stride(self):
        # Тестирование свертки с использованием заданного шага (stride)
        input_tensor = np.random.rand(2, 3, 5, 5)
        weight = np.random.rand(4, 3, 3, 3)

        output_tensor = convolution2D(input_tensor, weight, stride=2)

        # Утверждения для проверки корректности output_tensor
        self.assertEqual(output_tensor.shape, (2, 4, 2, 2))  # Пример размеров выходного тензора
        

    def test_convolution_with_padding(self):
        # Тестирование свертки с использованием заданного отступа (padding)
        input_tensor = np.random.rand(2, 3, 4, 4)
        weight = np.random.rand(4, 3, 3, 3)

        output_tensor = convolution2D(input_tensor, weight, padding=1)

        # Утверждения для проверки корректности output_tensor
        self.assertEqual(output_tensor.shape, (2, 4, 4, 4))  # Пример размеров выходного тензора с отступом
        

if __name__ == '__main__':
    unittest.main()
