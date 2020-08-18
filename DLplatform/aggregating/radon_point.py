import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from DLplatform.parameters import Parameters
import numpy as np
from DLplatform.aggregating import Aggregator


class RadonPoint(Aggregator):
    """
    Provides a method to aggregate n models using the Radon point method
    """

    RADON_CALC_METHOD = 1
    EPS = 0.000001
    MAX_REL_EPS = 0.0001

    def __init__(self, name="Radon point"):
        """

                        Returns
                        -------
                        None
            """
        Aggregator.__init__(self, name=name)

    # Added to mimic parallel classes (GeometricMedian, Average, etc. Remove if unnecessary)
    def calculateDivergence(self, param1, param2):
        if type(param1) is np.ndarray:
            return np.linalg.norm(param1 - param2) ** 2
        else:
            return param1.distance(param2) ** 2

    def getRadonPoint(S):
        alpha = []
        if RADON_CALC_METHOD == 1:
            A = np.vstack((np.transpose(S), np.ones(S.shape[0])))
            z = np.zeros(S.shape[0])
            z[0] = 1.0
            A = np.vstack((A, z))
            b = np.zeros(S.shape[0])
            b[-1] = 1.0
            alpha = np.linalg.lstsq(A, b)[0]
        else:
            # log(S)
            A = S[:-1]
            # log(A)
            A = np.vstack((np.transpose(A), np.ones(A.shape[0])))
            # log(A)
            b = np.hstack((S[-1], np.ones(1)))
            # log(b)
            alpha = np.linalg.solve(A, b)
        alpha_plus = np.zeros(len(alpha))
        alpha_minus = np.zeros(len(alpha))
        for i in range(len(alpha)):
            if alpha[i] > 0:
                alpha_plus[i] = alpha[i]
            if alpha[i] < 0:
                alpha_minus[i] = alpha[i]
        sumAlpha_plus = 1. * np.sum(alpha_plus)
        sumAlpha_minus = -1. * np.sum(alpha_minus)
        if not floatApproxEqual(sumAlpha_plus, sumAlpha_minus):
            pass
            # log("Error: sum(a+) != sum(a-): " + str(sumAlpha_plus) + " != " + str(sumAlpha_minus) + " for |S| = " + str(
            #     S.shape) + " and R = " + str(getRadonNumber(S)))
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
        if np.linalg.norm(rtest_plus + rtest_minus) > EPS:
            pass
            # log("Something went wrong!!! r+ = " + str(r) + " but r- = " + str(-1 * r_minus) + ". They should be the same!")
        return r

    def getRadonPointHierarchical(S, h):
        instCount = S.shape[0]
        if instCount == 1:
            return S[0]
        R = getRadonNumber(S)
        oldh = h
        if R ** h != instCount:
            # log("Unexpected number of points received for height " + str(h) + ".")
            pass
        while R ** h > instCount:
            h -= 1
        if oldh != h:
            # log("Had to adapt height to " + str(h) + ".")
            pass
        S = S[: int((instCount / R) * R)]  # ensures that |S| mod sampleSize == 0
        if (instCount / R) * R == 0:
            # log(R)
            # log(h)
            # log(instCount)
            # return getRadonPointRec(S,R)
            pass
        return getRadonPointIter(S, R)

    def getRadonPointRec(S, R):
        S_new = []
        if S.shape[0] == 1:
            return S[0]
        if S.shape[0] < R:
            # log("Error: too few instances in S for radon point calculation! |S| = " + str(
            #     S.shape[0]) + " for radon number R = " + str(R) + " .")
            return S[0]
        executor = ThreadPoolExecutor(max_workers=R)
        futures = []
        for i in range(S.shape[0] / R):
            futures.append(executor.submit(getRadonPoint(S[i * R:(i + 1) * R])))
        for f in as_completed(futures):
            S_new.append(f.result)
        gc.collect()
        return getRadonPointRec(np.array(S_new), R)

    def getRadonPointIter(S, R):
        while S.shape[0] >= R:
            S_new = []
            with ThreadPoolExecutor(max_workers=R) as executor:
                futures = []
                for i in range(int(S.shape[0] / R)):
                    futures.append(executor.submit(getRadonPoint, S[i * R:(i + 1) * R]))
                for f in as_completed(futures):
                    S_new.append(f.result())
            S = np.array(S_new)
            gc.collect()
        if S.shape[0] > 1:
            # log("Error: too few instances in S for radon point calculation! |S| = " + str(
            #     S.shape[0]) + " for radon number R = " + str(R) + " .")
            pass
        return S[0]

    def getRadonNumber(S):
        return S.shape[1] + 2  # for Euclidean space R^d the radon number is R = d + 2

    def floatApproxEqual(x, y):
        if x == y:
            return True
        relError = 0.0
        if (abs(y) > abs(x)):
            relError = abs((x - y) / y);
        else:
            relError = abs((x - y) / x);
        if relError <= MAX_REL_EPS:
            return True
        if abs(x - y) <= EPS:
            return True
        return False


if __name__ == "__main__":
    # Assume this represents 5 data points with 2 dimensions
    # np.random.seed(123)
    # # model_params_flat = np.random.randint(low=0, high=100, size=10)
    #
    # # Reshaping
    # # model_params = model_params_flat.reshape(5, 2)
    #
    model_params = np.array([[-2, 0], [0, 4], [4, 0], [0, 2]])
    #
    # rp = getRadonPoint(model_params)
    # print(rp)

    # S = np.array(
    #     [[2., 2.], [3., 1.], [3., 3.], [4., 4.], [2., 1.], [3., 1.], [3., 1.], [4., 4.], [2., 8.], [5., 3.], [3., 3.],
    #      [4., 4.], [1., 5.], [3., 7.], [3., 3.], [2., 4.]])
    r = getRadonPointHierarchical(model_params, 2)
    print(r)
