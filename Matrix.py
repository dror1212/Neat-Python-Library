import numpy as np


class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def set(self, mat):
        self.data = mat

    def randomize(self):
        self.data = np.random.uniform(-1.0, 1.0, (self.rows, self.cols))

    def multiply(self, num):
        try:
            self.data = np.multiply(self.data, num.data)
        except:
            self.data = np.multiply(self.data, num)

    def add(self, num):
        try:
            self.data = np.add(self.data, num.data)
        except:
            self.data = np.add(self.data, num)

    @staticmethod
    def sub(self, num):
        m = Matrix(self.rows, self.cols)
        try:
            m.data = np.subtract(self.data, num.data)
        except:
            m.data = np.subtract(self.data, num)
        return m

    def selfTranspose(self):
        self.data = self.data.transpose()

    @staticmethod
    def algebra_multiply(another_mat, mat):
        result = Matrix(another_mat.rows, mat.cols)
        result.data = np.matmul(another_mat.data, mat.data)

        return result

    @staticmethod
    def transpose(mat):
        result = Matrix(mat.rows, mat.cols)
        result.data = mat.data.transpose()
        return result

    @staticmethod
    def fromArray(arr):
        mat = Matrix(len(arr), 1)
        mat.data = np.asmatrix(arr).transpose()
        return mat

    def toArray(self):
        return np.squeeze(np.asarray(self.data))

    def map(self, func):
        self.data = np.vectorize(func)(self.data)

    def VariableMap(self, func, other):
        self.data = np.vectorize(func)(self.data, other.data)

    def BigMap(self, func, other, next):
        self.data = np.vectorize(func)(self.data, other, next)

    @staticmethod
    def staticMap(self, func):
        mat = Matrix(self.rows, self.cols)
        mat.data = self.data
        mat.map(func)
        return mat

    def copy(self):
        mat = Matrix(self.rows, self.cols)
        mat.data = self.data.copy()

        return mat
