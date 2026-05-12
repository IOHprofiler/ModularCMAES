"""This class creates the object problem. CEC'2013 test problems have already been defined with ID=1,...,20. It is possible to add custom problems
which is shown by one example.

Written by Ali Ahrari (aliahrari1983@gmail.com)
last updated by Ali Ahrari on 11 Jan 2021"""

import numpy as np


class BasicFun:
    @staticmethod
    def evaluate(x, hGO, funID):
        if funID == 1:
            f = BasicFun.elliptic(x, hGO)
        elif funID == 2:
            f = BasicFun.diffPow(x, hGO)
        elif funID == 3:
            f = BasicFun.schwefelN02Skew(x, hGO)
        elif funID == 4:
            f = BasicFun.rosenbrock(x, hGO)
        elif funID == 5:
            f = BasicFun.ackleySkew(x, hGO)
        elif funID == 6:
            f = BasicFun.rastrigin(x, hGO)
        elif funID == 7:
            f = BasicFun.weierstrass(x, hGO)
        elif funID == 8:
            f = BasicFun.schwefelN26(x, hGO)
        else:
            raise Exception("This function is not defined")
        return f

    def ackleySkew(x, hGO):
        y = 5.0 ** (np.sign(x) * hGO) * x
        term1 = np.sqrt(np.mean(y**2))
        term2 = np.mean(np.cos(2 * np.pi * y))
        f = -20 * np.exp(-0.2 * term1) - np.exp(term2) + 20 + np.exp(1)
        return f

    def diffPow(x, hGO):
        D = x.size
        H = hGO * 4.0
        p = 2.0 + H * np.arange(D) / (D - 1)
        f = np.sum(np.abs(x) ** p) ** 0.5
        return f

    def elliptic(x, hGO):
        D = x.size
        pow0 = np.arange(D) / (D - 1) * hGO * 3.0
        f = 10000.0 ** (0.5 - hGO) * np.sum((10.0**pow0 * x) ** 2)
        return f

    def rosenbrock(x, hGO):
        fsphere = 20 * np.sum(x**2)
        y = x + 1
        term1 = 100 * np.sum((y[1:] - y[0:-1] ** 2) ** 2)
        term2 = np.sum((y[0:-1] - 1) ** 2)
        f0 = term1 + term2
        f = hGO * f0 + (1 - hGO) * fsphere
        return f

    def rastrigin(x, hGO):
        base = 5
        A = 10.0 * (base**hGO - 1) / (base - 1)
        f = (x**2) + (A * (1 - np.cos(2 * np.pi * x)))
        f = np.sum(f)
        return f

    def schwefelN02Skew(x, hGO):
        D = x.size
        y = 5.0 ** (np.sign(x) * hGO) * x
        f0 = 0
        for k in np.arange(D):
            f0 = f0 + np.sum(y[0 : k + 1]) ** 2
        f = f0
        return f

    def schwefelN26(x, hGO):
        D = x.size
        base = 5.0
        H = (base**hGO - 1) / (base - 1)
        xstar = 420.96874635998202731184436501869
        # from matlab vpasolve
        fshift = 418.9828872724337062747864351956
        y = x + xstar
        p1 = np.sum((-300 - y) * (y < -500))
        p2 = np.sum((y > 500) * (y - 420))
        P = p1 + p2
        g = 1.5 * P + np.sum(-y * np.sin(np.abs(y) ** 0.5)) + fshift * D
        f = H * g + 1 * (1 - H) * np.sum(np.abs(y - xstar))
        return f

    def weierstrass(x, hGO):
        base = 5.0
        H = (base**hGO - 1) / (base - 1)
        D = x.size
        a = 0.5
        # controls the global basin (a higher a makes problem harder)
        b = 3.0
        # a higher value makes it makes it more rugged-default is 3
        k = np.arange(21, dtype=float)
        # level of optima
        term1 = (2 * np.pi * b**k).reshape(-1, 1) @ np.atleast_2d(x + 0.5)
        # size=21xD
        h1 = np.atleast_2d(a**k) @ np.cos(term1)
        h2 = np.sum(a**k * np.cos(np.pi * b**k))
        P = np.sum((np.abs(x) - 0.5) ** 1 * (np.abs(x) > 0.5))
        f0 = np.sum(h1) - D * h2 + P
        f = f0 * H + (1 - H) * np.sum(np.abs(x))
        return f


if __name__ == "__main__":  # a simple test of this class
    f = BasicFun.evaluate(np.array([2, 3]), 1, 7)
    print(f)
