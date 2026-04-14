import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# np.random.seed(7)
import matplotlib



def random_mask(input_size, mask_ratio):

    height = width = input_size
    num_patches = int(height * width)
    num_mask = int(num_patches * mask_ratio)

    mask = np.hstack([
            np.ones(num_patches - num_mask),
            np.zeros(num_mask),
    ])
    np.random.shuffle(mask)
    mask = mask.reshape([height, width])
    return mask


mask_point = random_mask(20,0.3)
plt.imshow(mask_point)
plt.show()



def block_mask(input_size, mask_ratio):
    """
    创建一个指定形状的零矩阵，然后将该矩阵中50%的元素置1，并保证这些被置1的元素相互紧邻。
    :param shape: 矩阵的形状，如(10, 10)表示创建一个10x10的矩阵。
    :return: 矩阵中50%的元素置1，并保证这些被置1的元素相互紧邻后的结果。
    """
    # 创建一个指定形状的零矩阵
    matrix = np.ones([input_size,input_size])

    # 计算需要设置为1的元素个数
    num_ones = int(matrix.size * mask_ratio)

    # 随机选定一个起始位置
    start = np.random.randint(0, matrix.size)

    # 设置起始位置为1
    matrix.flat[start] = 1

    # 初始化一个已经设置为1的元素的列表
    ones = [start]

    # 在已设置为1的元素周围寻找未设置为1的元素，直到达到需要设置的元素数量
    while len(ones) < num_ones:
        # 随机从已设置为1的元素列表中选择一个元素
        i = np.random.choice(ones)

        # 获取该元素在矩阵中的行和列索引
        row, col = np.unravel_index(i, matrix.shape)

        # 在该元素周围寻找未设置为1的元素
        for r in range(row-1, row+2):
            for c in range(col-1, col+2):
                # 跳过越界的元素和已经设置为1的元素
                m=r*matrix.shape[0]+c
                if r < 0 or r >= matrix.shape[0] or c < 0 or c >= matrix.shape[1] or m in ones:
                    continue

                # 将未设置为1的元素中的50%设置为1
                if np.random.random() < 0.5:
                    index = np.ravel_multi_index((r, c), matrix.shape)
                    matrix.flat[index] = 0
                    ones.append(index)

                # 如果已经达到需要设置的元素数量，直接返回结果矩阵
                if len(ones) == num_ones:
                    return matrix
    return matrix



mask_block = block_mask(20,0.3)
plt.imshow(mask_block)
plt.show()


