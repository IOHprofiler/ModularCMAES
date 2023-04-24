import unittest
import numpy as np
from itertools import combinations
from sklearn.svm import SVC


def extract_svm_coefs(svm: SVC, targets: list[str], target: str):
    return [(svm.coef_[i], svm.intercept_[i]) for i, c in enumerate(combinations(targets, 2)) if target in c]

def get_intersections(centroid: np.ndarray, coefs: list, point: np.ndarray):
    intersections = []
    for coef_, intercept_ in coefs:
        A = [np.append(coef_, 0)]
        B = [-intercept_]
        t = centroid - point
        for i in range(len(t)):
            x = np.zeros((len(t) + 1))
            x[i] = 1
            x[-1] = t[i]
            A.append(x)
            B.append(point[i])
        intersections.append(np.linalg.solve(A, B)[:-1])
    on_segment = []
    for i, intersection in enumerate(intersections):
        if (
            abs(np.linalg.norm(intersection - point)) > 1e-10
            and abs(
                np.linalg.norm(centroid - point)
                - (
                    np.linalg.norm(centroid - intersection)
                    + np.linalg.norm(intersection - point)
                )
            )
            < 1e-10
        ):
            on_segment.append(i)
    return on_segment, intersections

def closest_intersect(
    centroid: np.ndarray, on_segment: list[int], intersections: list[np.ndarray]
):
    distances = [np.linalg.norm(intersections[i] - centroid) for i in on_segment]
    return on_segment[np.argmin(distances)]

def ortho_projection(
    centroid: np.ndarray,
    coefs: list,
    point: np.ndarray,
    predict: SVC.predict,
    target: int,
):
    on_segment, intersections = get_intersections(
        centroid=centroid, point=point, coefs=coefs
    )
    if len(on_segment) == 0:
        return point
    j = closest_intersect(
        centroid=centroid, on_segment=on_segment, intersections=intersections
    )
    plane = coefs[j][0]
    vector = point - intersections[j]
    # projection
    pj = -np.dot(plane, vector) / np.dot(plane, plane)
    pj_point = plane * pj + point
    if (
        predict([pj_point])[0] != target
        or len(get_intersections(centroid=centroid, point=pj_point, coefs=coefs)[0]) > 0
    ):
        return ortho_projection(
            centroid=centroid,
            coefs=coefs,
            point=pj_point,
            target=target,
            predict=predict,
        )
    return pj_point

class TestOrthogonalProjection2d(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(7)
        lb, ub = -5, 5
        self.d = 2
        self.n = 4

        self.centroids = np.random.uniform(lb, ub, (self.n, self.d))
        self.svm = SVC(kernel='linear').fit(self.centroids, range(len(self.centroids)))
        
        self.new_point = np.random.uniform(lb, ub, (self.d,))

    def test_projection_2d(self):
        target = 2
        pj_point = ortho_projection(centroid=self.centroids[target], coefs=extract_svm_coefs(self.svm, range(self.n), target), point=self.new_point, predict=self.svm.predict, target=target)

        self.assertTrue(np.linalg.norm(pj_point - [1.626591178605456,-1.161121203257251]) < 1e-8)


class TestOrthogonalProjection3d(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        lb, ub = -5, 5
        self.d = 3
        self.n = 4

        self.centroids = np.random.uniform(lb, ub, (self.n, self.d))
        self.svm = SVC(kernel='linear').fit(self.centroids, range(len(self.centroids)))
        
        self.new_point = np.random.uniform(lb, ub, (self.d,))

    def test_projection_3d(self):
        target = 2
        pj_point = ortho_projection(centroid=self.centroids[target], coefs=extract_svm_coefs(self.svm, range(self.n), target), point=self.new_point, predict=self.svm.predict, target=target)

        self.assertTrue(np.linalg.norm(pj_point - [-3.30667606, -2.73953197,  1.40188444]) < 1e-8)

if __name__ == "__main__":
    unittest.main()
