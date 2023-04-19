import unittest
import numpy as np
from itertools import combinations
from sklearn.svm import SVC


def extract_svm_coefs(svm: SVC, targets: list[str], target: str):
    return [(svm.coef_[i], svm.intercept_[i]) for i, c in enumerate(combinations(targets, 2)) if target in c]

def ortho_projection(centroid: np.ndarray, coefs: list, point: np.ndarray):
        intersections = []
        for coef_, intercept_ in coefs:
            f = np.append(coef_, 0)
            # solve for vector intercept
            A = [f]
            B = [-intercept_]
            t = centroid - point
            for i in range(len(t)):
                x = np.zeros((len(t)+1,))
                x[i] = 1
                x[-1] = t[i]
                A.append(x)
                B.append(point[i])
            intersection = np.linalg.solve(A, B)[:-1]
            intersections.append(intersection)
        valids = []
        for i, intersection in enumerate(intersections):
            if abs(np.linalg.norm(centroid - point) - (np.linalg.norm(centroid - intersection) + np.linalg.norm(intersection - point))) < 1e-10:
                valids.append(i)
        distances = [np.linalg.norm(intersections[i] - centroid) for i in valids]
        j = valids[np.argmin(distances)]
        plane = coefs[j][0]
        vector = point - intersections[j]
        # projection
        pj = -np.dot(plane, vector) / np.dot(plane, plane)
        pj_point = plane*pj + point
        return intersections[j], pj_point

class TestOrthogonalProjection2d(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(7)
        lb, ub = -5, 5
        self.d = 2
        self.n = 3

        self.centroids = np.random.uniform(lb, ub, (self.n, self.d))
        self.svm = SVC(kernel='linear').fit(self.centroids, range(len(self.centroids)))
        
        self.new_point = np.random.uniform(lb, ub, (self.d,))

    def test_projection_2d(self):
        intersection, pj_point = ortho_projection(self.centroids[2], extract_svm_coefs(self.svm, range(self.n), 2), self.new_point)

        self.assertTrue(all(np.round(intersection, 8) == [1.62017982, -0.03736781]))
        self.assertTrue(all(np.round(pj_point, 8) == [0.77937692, -2.49010264]))


class TestOrthogonalProjection3d(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(123)
        lb, ub = -5, 5
        self.d = 3
        self.n = 3

        self.centroids = np.random.uniform(lb, ub, (self.n, self.d))
        self.svm = SVC(kernel='linear').fit(self.centroids, range(len(self.centroids)))
        
        self.new_point = np.random.uniform(lb, ub, (self.d,))

    def test_projection_3d(self):
        intersection, pj_point = ortho_projection(self.centroids[2], extract_svm_coefs(self.svm, range(self.n), 2), self.new_point)

        self.assertTrue(all(np.round(intersection, 8) == [2.66827713,  1.63606552, -0.76922988]))
        self.assertTrue(all(np.round(pj_point, 8) == [2.68175491,  0.97702187, -1.26411226]))

if __name__ == "__main__":
    unittest.main()
