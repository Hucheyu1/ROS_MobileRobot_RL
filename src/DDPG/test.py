import torch

# 假设 x 是一个 5x3x8 的张量
# x = torch.tensor([[[1, 2, 3, 4, 1, 2, 3, 4],
#                    [5, 6, 7, 8, 1, 2, 3, 4],
#                    [9, 10, 11, 12, 1, 2, 3, 4]],

#                   [[13, 14, 15, 16, 1, 2, 3, 4],
#                    [17, 18, 19, 20, 1, 2, 3, 4],
#                    [21, 22, 23, 24, 1, 2, 3, 4]],

#                   [[25, 26, 27, 28, 1, 2, 3, 4],
#                    [29, 30, 31, 32, 1, 2, 3, 4],
#                    [33, 34, 35, 36, 1, 2, 3, 4]],

#                   [[37, 38, 39, 40, 1, 2, 3, 4],
#                    [41, 42, 43, 44, 1, 2, 3, 4],
#                    [45, 46, 47, 48, 1, 2, 3, 4]],

#                   [[49, 50, 51, 52, 1, 2, 3, 4],
#                    [53, 54, 55, 56, 1, 2, 3, 4],
#                    [57, 58, 59, 60, 1, 2, 3, 4]]])
# len = torch.tensor([1,2,3,4,5])
# len = (len - 1).view(-1, 1).repeat(1, 8).unsqueeze(1).long()
# print(len)
# x = torch.gather(x, 0 ,len)
# print(x)
# print(x.squeeze(1))
input = torch.randn(10, 5, 20)
print(input)
# 创建索引张量 index，形状为 (64, 1, 256)
index = torch.randint(0, 10, size=[10,1,5])
print(index)
# 使用 torch.gather 函数收集元素
output = torch.gather(input, 1, index)
print(output)