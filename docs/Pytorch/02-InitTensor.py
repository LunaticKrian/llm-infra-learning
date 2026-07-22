import torch

# 基于数据创建一个张量Tensor
tensor_data_number = torch.tensor(10.0)
print(f"tensor_data: {tensor_data_number}")
print(f"tensor_type: {tensor_data_number.dtype}")
print(f"tensor_shap: {tensor_data_number.shape}")

# 基于一个列表创建一个张量Tensor
tensor_data_array = torch.tensor([10.0, 20.0, 30.0])
print(f"tensor_data: {tensor_data_array}")
print(f"tensor_type: {tensor_data_array.dtype}")
print(f"tensor_shap: {tensor_data_array.shape}")

# 基于ndarray创建tensor
import numpy as np

ndarray_data = np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
tensor_data_ndarray = torch.tensor(ndarray_data)
print(f"tensor_data: {tensor_data_ndarray}")
print(f"tensor_type: {tensor_data_ndarray.dtype}")
print(f"tensor_shap: {tensor_data_ndarray.shape}")

# ---

tensor_shape_size = torch.Tensor(2, 3, 2)
print(f"tensor_shape_size: {tensor_shape_size}")
print(f"tensor_shape_size: {tensor_shape_size.dtype}")
print(f"tensor_shape_size: {tensor_shape_size.shape}")

# 创建指定数据类型的Tensor
tensor_data_int = torch.IntTensor(2, 3, 2)
print(f"tensor_data_int: {tensor_data_int}")
print(f"tensor_data_int: {tensor_data_int.dtype}")
print(f"tensor_data_int: {tensor_data_int.shape}")

tensor_data_float = torch.FloatTensor(2, 3, 2)
print(f"tensor_data_float: {tensor_data_float}")
print(f"tensor_data_float: {tensor_data_float.dtype}")
print(f"tensor_data_float: {tensor_data_float.shape}")

tensor_int_data = torch.tensor([1, 2, 3], dtype=torch.int64)
print(f"tensor_int_data: {tensor_int_data}")
print(f"tensor_int_data: {tensor_int_data.dtype}")
print(f"tensor_int_data: {tensor_int_data.shape}")

# 创建指定区间张量数据
tensor_data_arange = torch.arange(1, 10, 2)
print(f"tensor_data_arange: {tensor_data_arange}")
print(f"tensor_data_arange: {tensor_data_arange.dtype}")
print(f"tensor_data_arange: {tensor_data_arange.shape}")

# 按照元素区间指定元素数量创建张量
tensor_linspace = torch.linspace(1, 10, 3)
print(f"tensor_linspace: {tensor_linspace}")
print(f"tensor_linspace: {tensor_linspace.dtype}")
print(f"tensor_linspace: {tensor_linspace.shape}")

# 生成对数Tensor
tensor_data_log = torch.logspace(1, 3, 2, 10)
print(f"tensor_data_log: {tensor_data_log}")
print(f"tensor_data_log: {tensor_data_log.dtype}")
print(f"tensor_data_log: {tensor_data_log.shape}")

# 创建一个全零张量
tensor_zero = torch.zeros(1, 3, 2)
print(f"tensor_zero: {tensor_zero}")
print(f"tensor_zero: {tensor_zero.dtype}")
print(f"tensor_zero: {tensor_zero.shape}")

tensor_zero_size = torch.zeros_like(tensor_zero)
print(f"tensor_zero_size: {tensor_zero_size}")
print(f"tensor_zero_size: {tensor_zero_size.dtype}")
print(f"tensor_zero_size: {tensor_zero_size.shape}")

# 创建一个单位矩阵
tensor_eye = torch.eye(3)
print(f"tensor_eye: {tensor_eye}")
print(f"tensor_eye: {tensor_eye.dtype}")
print(f"tensor_eye: {tensor_eye.shape}")

tensor_eye_m = torch.eye(3, 2)
print(f"tensor_eye_m: {tensor_eye_m}")
print(f"tensor_eye_m: {tensor_eye_m.dtype}")
print(f"tensor_eye_m: {tensor_eye_m.shape}")