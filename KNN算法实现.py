# 第一周任务：手撸基本knn算法，数据集使用sklearn自带数据集，实现过程仅允许使用numpy思考knn算法可以如何拓展
# 附加题:如何确定k的最佳取值

from sklearn.datasets import load_iris  # 从sklearn引入数据
import numpy as np
import random

iris_dataset = load_iris()

# 分组
n = len(iris_dataset['data']) // 10
data = iris_dataset['data']
target = iris_dataset['target']

b = np.array(target)
a = np.array(data)
A = []
i = 0
while i < 10 * n:
    mid = np.append(a[i], b[i])
    A = np.append(A, mid)
    i += 1

data = A.reshape(150, 5)
random.shuffle(data)
test_set = data[0:n]
train_set = data[n:]


# knn算法实现部分
# 1.算距离
# 2.升序排列
# 3.取前k个
# 4.加权平均

# 算欧几里得距离
def distance(d1, d2):
    res = 0
    j = 0
    m = [[1], [1], [1], [1], [0]]  # 线性代数运算，1*5的向量*5*1的向量，去除了最后的数字
    res = (d1 - d2) ** 2
    res = np.dot(res, m)
    return res ** 0.5


# 具体knn算法
def knn(data1):
    res = [
        {"result": 'train[4]', "distance": distance(data1, train)}
        for train in train_set
    ]

    res = sorted(res, key=lambda item: item['distance'])  # 将计算结果顺序排列
    res2 = res[0:K]  # 取了前k个元素
    # 确定权重
    sum1 = 0
    for r in res2:
        sum1 += r['distance']
    return sum1
    for r in res2:
        result[r['result']] += 1 - r["diatance"] / sum
    if result['0.0'] > (result['1.0'] and result['2.0']):
        return 0.0
    elif result['1.0'] > (result['0.0'] and result['2.0']):
        return 1.0
    elif result['2.0'] > (result['1.0'] and result['2.0']):
        return 2.0


# 测试阶段
K = 3  # 确定k数值
correct = 0
for test in test_set:
    result = test[4]
    result2 = knn(test)
    if result == result2:
        correct += 1

print("准确率：{:.2f}%".format(100 * correct / len(test_set)))
