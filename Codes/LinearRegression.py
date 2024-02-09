
import random
random.seed(1229)

# Linear Regression with gradient descent
class Linear_Regression():
    def __init__(self):
        # 파라미터 초기화
        self.weight = random.random()
        self.bias = 0
        self.learning_rate = 0.0001

    def forward(self, train_input):
        prediction = train_input * self.weight + self.bias # y = ax + b
        return prediction

    def backward(self, train_input, prediction, train_target):
        # 오차 계산(mse)
        errors = (train_target - prediction)**2

        # 각 파라미터에 에 대한 grad 계산
        gradient_weight = 2 * errors * train_input
        gradient_bias = 2 * errors

        # 가중치 업데이트
        self.weight -= self.learning_rate * gradient_weight
        self.bias -= self.learning_rate * gradient_bias

    def train(self, train_input, train_target, epochs):
        for i in range(epochs):
          for j in range(len(train_input)):
            # stochastic gradient descent
            prediction = self.forward(train_input[j])
            self.backward(train_input[j], prediction, train_target[j])

    def test(self, test_input, test_target):
        acc = self.score(test_input, test_target)
        print(acc)

    def score(self, train_input, train_target):
        pred = train_input * self.weight + self.bias
        acc = ((train_target - pred) ** 2).mean()
        return acc

#데이터 불러오기
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

import numpy as np

bmi_index = 2

# 데이터셋 불러오기
from sklearn.model_selection import train_test_split

diabetes_data = diabetes.data[:, bmi_index]
diabetes_target = diabetes.target

# 결측치 확인
nan_indices = np.isnan(diabetes_data)
diabetes_data_no_nan = diabetes_data[~nan_indices]
diabetes_target_no_nan = diabetes_target[~nan_indices]

train_input, test_input, train_target, test_target = train_test_split(diabetes_data_no_nan, diabetes_target_no_nan,
                                                                      test_size=0.3, shuffle=True, random_state=42)

LR = Linear_Regression()
LR.train(train_input, train_target, 5)
LR.score(train_input, train_target)
LR.test(test_input, test_target)

