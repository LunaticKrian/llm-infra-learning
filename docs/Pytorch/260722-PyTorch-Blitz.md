from torch._dynamo.variables import torchfrom sympy.stats.sampling.sample_numpy import numpy

# PyTorch Blitz

https://docs.pytorch.org/tutorials/beginner/blitz/index.html

**Mac PyTorch 安装：** 官方站点 https://pytorch.org/

```bash
pip3 install torch
```

---

## Tensors

https://docs.pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

> 学习思想，理解概念，记 API 没有意义（投入产出比太低）～
> 
> Learning theories, understanding concepts, and memorizing APIs make no sense (the return on investment is too low) ~

### Tensor Initialization 

Tensor 初始化可以通过多种方式实现

#### 基于 data 初始化

直接基于给定的数据进行初始化，初始化的Tensor中的数据类型是基于原始数据进行自动推断

```python
import torch

data = [[1, 2], [3, 4]]
x = torch.tensor(data)
```

#### 基于 Numpy 初始化

Tensor 可以基于Numpy数组进行数据初始化，Tensor会保留原NumPy数组的数据类型

```python
import torch
import numpy

data = [[1, 2], [3, 4]]
np_array = numpy.array(data)
x = torch.from_numpy(np_array)
```

#### 基于 Tensor 初始化

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

#### 基于 随机/固定值 初始化

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

### Tensor Attributes

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

### Tensor Operations

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