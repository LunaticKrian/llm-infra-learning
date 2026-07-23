# Tensors

https://docs.pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

> 学习思想，理解概念，记 API 没有意义（投入产出比太低）～
> 
> Learning theories, understanding concepts, and memorizing APIs make no sense (the return on investment is too low) ~

## Tensor Initialization 

Tensor 初始化可以通过多种方式实现

### 基于 data 初始化

直接基于给定的数据进行初始化，初始化的Tensor中的数据类型是基于原始数据进行自动推断

```python
import torch

data = [[1, 2], [3, 4]]
x = torch.tensor(data)
```

### 基于 Numpy 初始化

Tensor 可以基于Numpy数组进行数据初始化，Tensor会保留原NumPy数组的数据类型

```python
import torch
import numpy

data = [[1, 2], [3, 4]]
np_array = numpy.array(data)
x = torch.from_numpy(np_array)
```

### 基于 Tensor 初始化

新创建的 Tensor 会保留作为参数传递进入的 Tensor 的属性（shape，datatype），除非显示声明会覆盖默认值

```python
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
```

输出：

```txt
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.9496, 0.7472],
        [0.2991, 0.5063]])
```

### 基于 随机/固定值 初始化

`shape`：是一个 tuple（元组）数据类型，声明 Tensor 的维度属性，决定了输出 Tensor 的数据维度。

```python
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

输出：

```text
Random Tensor:
 tensor([[0.2294, 0.5491, 0.9405],
        [0.2509, 0.6368, 0.2384]])

Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

---

## Tensor Attributes

Tensor Attributes 属性描述了这个 Tensor 的 shape（形状）、datatype（数据类型）、device（数据存储在什么设备上，CPU（内存）、GPU（显存））

```python
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

输出：

```text
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

---

## Tensor Operations

https://docs.pytorch.org/docs/2.13/torch.html

尝试把 Tensor 移动到GPU上进行后续计算：

```python
# We move our tensor to the GPU if available
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
tensor = tensor.to(device)
print(f"Device tensor is stored on: {tensor.device}")
```

> PS：`.to(device)` 这个操作本身不会改变后续所有代码的执行设备。它只移动了调用它的那个特定张量或模型。要确保计算在GPU上运行，需要对这个计算过程中用到的每一个张量都进行妥善的设备管理

输出：

```text
Device tensor is stored on: cuda:0
```

### Tensor 索引/切片

类似于Numpy的索引和切片

```python
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)
```

输出：

```text
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

### Tensor 拼接

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

输出：

```text
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```

### Tensor 乘法

PS：注意，这里是两个矩阵对应元素相乘

```python
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")
```

输出：

```text
tensor.mul(tensor)
 tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor * tensor
 tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

计算两个矩阵相乘（使用矩阵乘法）：

```python
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")
```

输出：

```text
tensor.matmul(tensor.T)
 tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])

tensor @ tensor.T
 tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
```

### In-place 原地操作

带有 _ 后缀的操作是原地操作。例如：x.copy_(y)，x.t_()，会修改 x

```python
print(tensor, "\n")
tensor.add_(5)
print(tensor)
```

输出：

```text
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```

> In-place 原地操作可以节省一些内存，但在计算导数时可能会因立即丢失历史信息而出现问题，因此不建议使用。

---

## Tensor 关联 Numpy

CPU上的 Tensor 张量和 NumPy 数组 可以共享其底层的内存位置，修改其中一个会同时影响另一个。

### Tensor 转换为 Numpy 数组

```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

输出：

```text
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
```

在 Tensor 中的变更会反应到 Numpy数组中（操作了同一内存中的数据）

```python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

输出：

```text
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```

### Numpy 数组 转换为 Tensor

```python
n = np.ones(5)
t = torch.from_numpy(n)
```

改变 Numpy 数组数据会影响 Tensor：

```python
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

输出：

```text
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
```
