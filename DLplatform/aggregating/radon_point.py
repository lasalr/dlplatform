import os
import random
from DLplatform.aggregating import Aggregator
from DLplatform.parameters import Parameters
from typing import List
import numpy as np
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

from DLplatform.parameters.vectorParameters import VectorParameter


class RadonPoint(Aggregator):
    """
    Provides a method to aggregate n models using the Radon point
    """

    EPS = 0.000001
    MAX_REL_EPS = 0.0001

    def __init__(self, name="Radon point"):
        """
        Returns
        -------
        None
        """
        Aggregator.__init__(self, name=name)

    def __str__(self):
        return "Radon point"

    def __call__(self, params: List[Parameters]) -> Parameters:
        """
        This aggregator takes n lists of model parameters and returns a Radon point of the model parameters
        parameters.

        Parameters
        ----------
        params A list of Parameter objects. These objects support addition and scalar multiplication.

        Returns
        -------
        A new parameter object that is the Radon point of params.

        """
        print('Running Radon point calculation with {} models with PID={}'.format(len(params), os.getpid()))
        counter = 0
        if len(params[1:]) == 0:
            print('WARNING: Only 1 model processed!')
        arr = self.get_array(params)
        return VectorParameter(self.getRadonPointHierarchical(vectors=arr))

    def floatApproxEqual(self, x, y):
        if x == y:
            return True
        relError = 0.0
        if (abs(y) > abs(x)):
            relError = abs((x - y) / y)
        else:
            relError = abs((x - y) / x)
        if relError <= self.MAX_REL_EPS:
            return True
        if abs(x - y) <= self.EPS:
            return True
        return False

    def get_dim(self, params):
        """
        Validates all models have same dimensionality
        and returns dimensionality
        """
        dim = params[0].dim
        for p in params:
            if p.dim != dim:
                raise ValueError("Number of parameters in each distributed model are not equal")

        return dim

    def get_array(self, params: List[Parameters]):
        """
        returns params as nd.array
        """
        arr = params[0].getCopy().get()
        # print('arr (at start) =', arr)
        if len(params[1:]) == 0:
            return arr.reshape((1, len(arr)))
        else:
            for p in params[1:]:
                # print('p =', p)
                arr = np.vstack((arr, p.getCopy().get()))
                # print('arr =', arr)
            return arr

    def getRadonPoint(self, S, r):
        alpha = []
        A = np.vstack((np.transpose(S), np.ones(S.shape[0])))
        z = np.zeros(S.shape[0])
        z[0] = 1.0
        A = np.vstack((A, z))
        b = np.zeros(S.shape[0])
        b[-1] = 1.0
        alpha = np.linalg.lstsq(A, b, rcond=None)[0]
        alpha_plus = np.zeros(len(alpha))
        alpha_minus = np.zeros(len(alpha))
        for i in range(len(alpha)):
            if alpha[i] > 0:
                alpha_plus[i] = alpha[i]
            if alpha[i] < 0:
                alpha_minus[i] = alpha[i]
        sumAlpha_plus = 1. * np.sum(alpha_plus)
        sumAlpha_minus = -1. * np.sum(alpha_minus)
        if not self.floatApproxEqual(sumAlpha_plus, sumAlpha_minus):
            print("Error: sum(a+) != sum(a-): " + str(sumAlpha_plus) + " != " + str(sumAlpha_minus) + " for |S| = " +
                  str(S.shape) + " and R = " + str(r))
        alpha /= sumAlpha_plus
        r = np.zeros(S.shape[1])
        r_minus = np.zeros(S.shape[1])
        for i in range(len(alpha)):
            if alpha[i] > 0:
                r += alpha[i] * S[i]
            if alpha[i] < 0:
                r_minus += alpha[i] * S[i]
        rtest_plus = r * 1. / np.linalg.norm(r)  # normiert
        rtest_minus = r_minus * 1. / np.linalg.norm(r_minus)  # normiert
        if np.linalg.norm(rtest_plus + rtest_minus) > self.EPS:
            print("Something went wrong!!! r+ = " + str(r) + " but r- = " + str(-1 * r_minus) +
                  ". They should be the same!")
        return r

    def getRadonPointHierarchical(self, vectors):
        n, d = vectors.shape
        r = d + 2
        h = math.floor(math.log(n, r))
        S = np.array(random.choices(vectors.tolist(), k=r ** h))
        print('n={}, d={}, r={}, h={}, r ** h={}, S.shape={}'.format(n, d, r, h, r ** h, S.shape))
        while S.shape[0] >= r:
            S_new = []
            print(S.shape[0] / r)
            for i in range(S.shape[0] // r):
                v = self.getRadonPoint(S[i * r:(i + 1) * r], r)
                S_new.append(v)
            S = np.array(S_new)
        if S.shape[0] > 1:
            print("Error: too few instances in S for radon point calculation! |S| = " + str(
                S.shape[0]) + " for radon number R = " + str(r) + " .")
        return S[0]



# if __name__ == "__main__":
#     # Assume this represents 5 data points with 2 dimensions
#     # np.random.seed(123)
#     # # model_params_flat = np.random.randint(low=0, high=100, size=10)
#     #
#     # # Reshaping
#     # # model_params = model_params_flat.reshape(5, 2)
#     #
#     model_params = np.array([[-2, 0], [0, 4], [4, 0], [0, 2]])
#     #
#     # rp = getRadonPoint(model_params)
#     # print(rp)
#
# # S = np.array( #     [[2., 2.], [3., 1.], [3., 3.], [4., 4.], [2., 1.], [3., 1.], [3., 1.], [4., 4.], [2., 8.],
# [5., 3.], [3., 3.], #      [4., 4.], [1., 5.], [3., 7.], [3., 3.], [2., 4.]]) r = getRadonPointHierarchical(
# model_params, 2) print(r)
