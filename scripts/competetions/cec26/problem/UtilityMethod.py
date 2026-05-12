# Some utility methods


import numpy as np
import copy
from scipy.spatial.distance import cdist


class UtilityMethod:
    @staticmethod
    def gen_rot_mat_pseudo(
        u0, v0, alp
    ):  # Generates a random rotation matrix given two random vectors uu and vv
        #  see [1] "On the Rigid Rotation Conept in n-Dimensional Spaces"  by
        # Daniele Mortari (2001) for mathematical formulation
        if not (np.ndim(u0) == 1 and np.ndim(v0) == 1 and np.ndim(alp) == 0):
            raise Exception("dimensions of inputs are incorrect")
        D = u0.size
        u = u0 / np.linalg.norm(u0)
        v = v0 - np.sum(u * v0) * u
        v = v / np.linalg.norm(v)
        u = np.atleast_2d(u).T
        v = np.atleast_2d(v).T
        R = (
            np.eye(D)
            + np.sin(alp) * (v @ u.T - u @ v.T)
            + (np.cos(alp) - 1) * (u @ u.T + v @ v.T)
        )
        return R

    def keep_farthest(X, n):
        # iteratively removes closest solutions from X such that n solutions remain in the end
        # each element of X is in [0,1]
        N, D = np.shape(X)
        dis = cdist(X, X)
        dis2 = dis + np.eye(N) * D**0.5 * 2
        # distance to self if very large
        keepInd = [0]
        candidInd = list(range(1, N))
        for k in np.arange(1, n):
            closest = np.min(np.atleast_2d(dis[np.ix_(keepInd, candidInd)]), axis=0)
            index = np.argmax(closest)  # index of farthest point from candidInd
            keepInd.append(candidInd[index])
            candidInd.pop(index)
        Y = copy.deepcopy(X[keepInd, :])
        minDis = np.min(dis2[np.ix_(keepInd, keepInd)])
        return Y, minDis

    def redist_glob_min(X, Xref, hardNU, disTol):
        # distance to the point Xref shrinks according to the rank of the solution
        # when all solutions are sorted according to their distance to Xref
        # disTol: Almost makes sure all solutions are at least disTol far from each other
        # 0 <= hardNU: Controls the nonlinearity of scaling
        maxTry = 100
        # maixumum number of tries so that each redictributed solution is far from the previously relocated solutions
        N, D = np.shape(X)
        countTry = np.zeros(N)
        tauNU = 0.2
        Y = copy.deepcopy(X)
        # redistributed solutions

        if N > 1:
            Vlength = np.zeros(N)
            V = np.zeros((N, D))
            for k in np.arange(N):
                V[k] = X[k] - Xref
                Vlength[k] = np.linalg.norm(V[k])

            ind = np.argsort(Vlength)
            rnk = np.zeros(N)
            rnk[ind] = np.arange(1, N + 1)
            # rank of each solution when sorted according to distance to Xref
            rnk = rnk / N * (1 - tauNU) + tauNU
            targetCoef = rnk**hardNU
            # reduce distance to Xref proportionally to the rank

            for k in np.arange(ind.size):
                for tryNo in np.arange(maxTry + 1):
                    term1 = (maxTry - tryNo) / maxTry
                    finCoef = targetCoef[ind[k]] ** term1
                    # final coefficient, ideally close to targetCoef unless it violates the disTol criterion
                    Y[ind[k], :] = Xref + finCoef * V[ind[k], :]
                    # now make sure Y(k,:) is at least disTol away from all previously
                    # redistributed solutions
                    tmp = np.delete(np.arange(N), ind[k])
                    dis1 = cdist(np.atleast_2d(Y[ind[k]]), np.atleast_2d(Y[tmp]))
                    countTry[ind[k]] = tryNo
                    # print(k,finCoef,np.min(dis1))

                    if (k == 0) or (k == N - 1) or (np.min(dis1) >= disTol):
                        break  # accept this new location
            # end: k-th solution has been relocated
        else:
            pass
        return Y, countTry

    """ Calculate the robust peak ratio given reported solutions X, their values  f, global minima (x_opt),
    global minimum value (f_opt), and the range of desired tolerance on the values """

    def calc_robust_peak_ratio(solutions, values, minima, minimum_val, ftol0):
        ftol = [np.min(ftol0), np.max(ftol0)]
        n_sol = solutions.shape[0]
        n_minima = minima.shape[0]
        pr = np.zeros(n_minima)
        cor_opt_ind = np.zeros(
            n_sol
        )  # index of corresponding global minima that this solution tried to approximate

        # find corresponding global minimum for each solution
        dis2opt = cdist(np.atleast_2d(solutions), minima)
        cor_opt_ind = np.argmin(dis2opt, axis=1)

        # calculate credit for approximating each global minimum
        for k in range(n_minima):
            ind = np.where(cor_opt_ind == k)[0].astype(int)
            if ind.size > 0:
                best_val = np.min(values[ind])
                if ftol[0] == ftol[1]:
                    pr[k] = (best_val <= (minimum_val + ftol[0])) * 1.0
                else:
                    term1 = np.log(ftol[1]) - np.log(best_val - minimum_val)
                    term2 = np.log(ftol[1]) - np.log(ftol[0])
                    pr[k] = np.max((0, np.min((1, term1 / term2))))
        return np.mean(pr), pr


if __name__ == "__main__":  # a simple test of this class
    from matplotlib import pyplot as plt

    R = UtilityMethod.gen_rot_mat_pseudo(np.arange(3), np.arange(2, 5), 0.3)
    print(R)
    np.random.seed(0)

    X = np.random.rand(100, 2)
    Y, tmp = UtilityMethod.keep_farthest(X, 10)

    plt.plot(X[:, 0], X[:, 1], "o")
    plt.plot(Y[:, 0], Y[:, 1], "xr")

    Z, temp = UtilityMethod.redist_glob_min(Y, Y[0], 3, 0.01)
    fig2 = plt.figure(2)
    ax = fig2.add_subplot(1, 1, 1)

    plt.plot(Y[:, 0], Y[:, 1], "xr")
    plt.plot(Y[0, 0], Y[0, 1], "or")

    plt.plot(Z[:, 0], Z[:, 1], "xg")
