import numpy as np
np.seterr(divide='ignore')

class NaiveBayes:

    # X - macierz [samplesNum x featuresNum]
    # y - ilosc probek
    def fit(self, X, y):
        samplesNum, featuresNum = X.shape
        self.classes = np.unique(y)
        classesNum = len(self.classes)
        self.eps = 1e-6

        # init wart. oczekiwana; wariancja; prior
        self._mean = np.zeros((classesNum, featuresNum), dtype=np.float64)
        self._var = np.zeros((classesNum, featuresNum), dtype=np.float64)
        self._priors = np.zeros(classesNum, dtype=np.float64)

        for index, c in enumerate(self.classes):
            X_c = X[y == c]  # wszystkie wektory cech gdzie label jest równa label y
            self._mean[index, :] = X_c.mean(axis=0)
            self._var[index, :] = X_c.var(axis=0)
            self._priors[index] = X_c.shape[0] / float(
                samplesNum)  # <- frakfencja (jak często klasa c pojawia się we wszystkich probkach)

    def predict(self, X):
        pred_y = [self._predict(x) for x in X]
        return pred_y

    def _predict(self, x):
        posteriors = []

        for index, c in enumerate(self.classes):
            prior = np.log(self._priors[index])
            classConditional = np.sum(np.log(self.probabiltyDensity(index, x)))
            posterior = prior + classConditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def probabiltyDensity(self, classIndex, x):  # liczone z rozkładu normalnego (Gaussa)
        mean = self._mean[classIndex]
        var = self._var[classIndex]
        licznik = np.exp(- (x - mean) ** 2 / (2 * var + self.eps))
        mianownik = np.sqrt(2 * np.pi * var) + self.eps
        return licznik / mianownik
