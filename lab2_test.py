import unittest
import torch
from lab2 import Convolution3DModule, Convolution3D

class TestConvolution3D(unittest.TestCase):     

    def test_convolution_3d_function(self):
        # Создаем входные данные и параметры
        input_tensor = torch.randn(1, 4, 4, 4, 3)  # Пример: батч размер 1, размеры 4x4x4, 3 канала
        weight_tensor = torch.randn(3, 3, 3, 3, 6)  # Пример: 3 фильтра размером 3x3x3 для каждого из 6 каналов
        bias_tensor = torch.randn(6)  # Пример: bias для каждого из 6 фильтров

        # Вызываем вашу функцию Convolution 3D
        output_tensor = Convolution3D(input_tensor, weight_tensor, bias_tensor, activation_fn=torch.nn.ReLU())

        # Проверяем ожидаемый размер выходного тензора или другие характеристики
        self.assertEqual(output_tensor.shape, torch.Size([1, 2, 2, 2, 6]))  # Пример: ожидаемый размер 1x2x2x2x6

        

    def test_convolution_3d_no_activation(self):
        # Тестирование Convolution3D без функции активации
        input_tensor = torch.randn(1, 4, 4, 4, 3)
        weight_tensor = torch.randn(3, 3, 3, 3, 6)
        bias_tensor = torch.randn(6)

        output_tensor = Convolution3D(input_tensor, weight_tensor, bias_tensor)

        self.assertEqual(output_tensor.shape, torch.Size([1, 2, 2, 2, 6]))

    def test_convolution_3d_invalid_input(self):
        # Тестирование Convolution3D с недопустимыми размерностями входных данных
        input_tensor = torch.randn(1, 4, 4, 3)  # Некорректная размерность
        weight_tensor = torch.randn(3, 3, 3, 3, 6)
        bias_tensor = torch.randn(6)

        with self.assertRaises(ValueError):
            Convolution3D(input_tensor, weight_tensor, bias_tensor)

if __name__ == '__main__':
    unittest.main()
