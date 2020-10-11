import numpy as np
from typing import List
from DLplatform.learning.learner import BatchLearner
from DLplatform.parameters.vectorParameters import VectorParameter
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import LinearSVC as SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.kernel_approximation import Nystroem

RANDOM_STATE = 123


class LinearSVC(BatchLearner):

    def __init__(self, regParam, dim, name="LinearSVC"):
        BatchLearner.__init__(self, name=name)
        self.regParam = regParam
        self.dim = dim
        self.model = SVC(C=self.regParam, loss='hinge', max_iter=2000, random_state=RANDOM_STATE)
        self.model.coef_ = np.zeros(dim - 1)
        self.model.intercept_ = np.array([0.0])

    def setModel(self, param: VectorParameter, setReference: bool):
        super(LinearSVC, self).setModel(param, setReference)

        # self.info('STARTTIME_setReference: '+str(time.time()))
        if setReference:
            self._flattenReferenceParams = param.get()
        # self.info('ENDTIME_setReference: '+str(time.time()))

    def train(self, data: List) -> List:
        """
        Training

        Parameters
        ----------
        data - training batch

        Returns
        -------
        list - first element is loss suffered on this training
                second element are predictions on the training data

        """

        if not isinstance(data, List):
            error_text = "The argument data is not of type" + str(List) + "it is of type " + str(type(data))
            self.error(error_text)
            raise ValueError(error_text)
        print('About to train using LinearSVC')
        X = np.asarray([record[0] for record in data])
        y = np.asarray([record[1] for record in data])

        clf = make_pipeline(StandardScaler(), self.model)
        clf.fit(X, y)
        # loss = self.model.fit(X=X, y=y).loss(X=X, y=y)
        loss = clf.fit(X=X, y=y).score(X=X, y=y)
        preds = clf.predict(X)

        return [loss, preds]

    def setParameters(self, param: VectorParameter):
        """

        Replace the current values of the model parameters with the values of "param"

        Parameters
        ----------
        param

        Returns
        -------

        Exception
        ---------
        ValueError
            in case that param is not of type Parameters
        """

        if not isinstance(param, VectorParameter):
            error_text = "The argument param is not of type" + str(VectorParameter) + "it is of type " \
                         + str(type(param))
            self.error(error_text)
            raise ValueError(error_text)
        # print('param.shape =', param.get().shape)
        w = param.get().tolist()
        # print('w =', w)
        b = w[-1]
        # print('b =', b)
        del w[-1]
        # print('w =', w)

        self.model.coef_ = np.array(w)
        self.model.intercept_ = np.array([b])
        # print('self.model.coef_ =', self.model.coef_)
        # print('self.model.intercept_ =', self.model.intercept_)

    def getParameters(self) -> VectorParameter:
        """

        Takes the current model parameters and hands them to a VectorParameter object which is returned

        Returns
        -------
        Parameters

        """

        # print('self.model.coef_ =', self.model.coef_)
        # print('self.model.intercept_ =', self.model.intercept_)
        # Flattened Coefficients of the support vectors followed by the intercepts
        wb = np.concatenate((self.model.coef_.flatten(), self.model.intercept_))
        # print('wb.shape =', wb.shape)
        # print('wb =', wb)
        # in principle, the intercept can be a list. But this may break at other points, then.
        if isinstance(self.model.intercept_, List):
            print('Intercept is a list')
            wb = np.array(self.model.coef_[0].tolist() + self.model.intercept_.tolist())
        return VectorParameter(wb)


class LogisticRegression(BatchLearner):

    def __init__(self, regParam, dim, solver='lbfgs', name="LogisticRegression"):
        BatchLearner.__init__(self, name=name)
        self.regParam = regParam
        self.dim = dim
        self.solver = solver
        self.model = LR(C=self.regParam, solver=self.solver, random_state=RANDOM_STATE)
        self.model.coef_ = np.zeros(dim - 1)
        self.model.intercept_ = np.array([0.0])
        # self.weights = np.zeros(dim)

    def setModel(self, param: VectorParameter, setReference: bool):
        super(LogisticRegression, self).setModel(param, setReference)

        # self.info('STARTTIME_setReference: '+str(time.time()))
        if setReference:
            self._flattenReferenceParams = param.get()
        # self.info('ENDTIME_setReference: '+str(time.time()))

    def train(self, data: List) -> List:
        '''
        Training

        Parameters
        ----------
        data - training batch

        Returns
        -------
        list - first element is loss suffered on this training
                second element are predictions on the training data

        '''

        if not isinstance(data, List):
            error_text = "The argument data is not of type" + str(List) + "it is of type " + str(type(data))
            self.error(error_text)
            raise ValueError(error_text)
        X = np.asarray([record[0] for record in data])
        y = np.asarray([record[1] for record in data])
        loss = self.model.fit(X, y).score(X, y)
        preds = self.model.predict(X)

        return [loss, preds]

    def setParameters(self, param: VectorParameter):
        '''

        Replace the current values of the model parameters with the values of "param"

        Parameters
        ----------
        param

        Returns
        -------

        Exception
        ---------
        ValueError
            in case that param is not of type Parameters
        '''

        if not isinstance(param, VectorParameter):
            error_text = "The argument param is not of type" + str(VectorParameter) + "it is of type " + str(
                type(param))
            self.error(error_text)
            raise ValueError(error_text)
        # TODO: so far, we assume that the intercept is a scalar, but it can be also a 1d-array with len > 1. This would have to be configured somehow...
        w = param.get().tolist()
        b = w[-1]
        del w[-1]
        self.model.coef_ = np.array(w)
        self.model.intercept_ = np.array([b])

    def getParameters(self) -> VectorParameter:
        '''

        Takes the current model parameters and hands them to a KerasNNParameters object which is returned

        Returns
        -------
        Parameters

        '''
        wb = np.concatenate((self.model.coef_.flatten(), self.model.intercept_))
        if isinstance(self.model.intercept_,
                      List):  # in principle, the intercept can be a lit. But this may break at other points, then.
            wb = np.array(self.model.coef_[0].tolist() + self.model.intercept_.tolist())
        return VectorParameter(wb)

    def calculateCurrentDivergence(self):
        return np.linalg.norm(self.getParameters().get() - self._flattenReferenceParams)


class LinearSVCRandomFF(LinearSVC):

    def __init__(self, regParam, dim, name="LinearSVCRandomFF"):
        super(LinearSVCRandomFF, self).__init__(regParam, dim, name)

    def train(self, data: List) -> List:
        """
        Training Linear SVC using random fourier features sampler

        Parameters
        ----------
        data - training batch

        Returns
        -------
        list - first element is loss suffered on this training
                second element are predictions on the training data

        """
        if not isinstance(data, List):
            error_text = "The argument data is not of type" + str(List) + "it is of type " + str(type(data))
            self.error(error_text)
            raise ValueError(error_text)
        print('About to train using LinearSVC with Random Fourier Features sampling')
        X = np.asarray([record[0] for record in data])
        y = np.asarray([record[1] for record in data])

        rff_sampler = RBFSampler(gamma=1, random_state=RANDOM_STATE)
        X_sampled = rff_sampler.fit_transform(X)

        clf = make_pipeline(StandardScaler(), self.model)
        fitted_model = clf.fit(X_sampled, y)
        # loss = self.model.fit(X=X, y=y).loss(X=X, y=y)
        loss = fitted_model.score(X=X_sampled, y=y)
        preds = clf.predict(X_sampled)

        return [loss, preds]


class LinearSVCNystrom(LinearSVC):

    def __init__(self, regParam, dim, name="LinearSVCNystrom"):
        super(LinearSVCNystrom, self).__init__(regParam, dim, name)

    def train(self, data: List) -> List:
        """
        Training Linear SVC using Nystrom sampler

        Parameters
        ----------
        data - training batch

        Returns
        -------
        list - first element is loss suffered on this training
                second element are predictions on the training data

        """
        if not isinstance(data, List):
            error_text = "The argument data is not of type" + str(List) + "it is of type " + str(type(data))
            self.error(error_text)
            raise ValueError(error_text)
        print('About to train using LinearSVC with Nystrom sampling')
        X = np.asarray([record[0] for record in data])
        y = np.asarray([record[1] for record in data])

        nystrom_sampler = Nystroem(gamma=1, random_state=RANDOM_STATE, n_components=int(len(X) * 0.1))
        X_sampled = nystrom_sampler.fit_transform(X)

        clf = make_pipeline(StandardScaler(), self.model)
        fitted_model = clf.fit(X_sampled, y)
        # loss = self.model.fit(X=X, y=y).loss(X=X, y=y)
        loss = fitted_model.score(X=X_sampled, y=y)
        preds = clf.predict(X_sampled)

        return [loss, preds]
