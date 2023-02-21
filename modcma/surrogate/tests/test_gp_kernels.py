
import unittest

from modcma.surrogate.gp_kernels import *



for kernel in basic_kernels + functor_kernels:
    locals()[kernel.__name__] = kernel

class Mock2:
    d = 2

class Mock5:
    d = 5

class TestKernels(unittest.TestCase):
    def test_dof(self):
        k = MaternOneHalf(Mock2)

        self.assertEqual(k.dof, 2)
        self.assertIsInstance(k.get_variables(), list)
        self.assertEqual(len(k.get_variables()), 2)

        class Mock5:
            d = 5
        k = MaternOneHalf
        k = k(Mock2)

        self.assertEqual(k.dof, 2)
        self.assertIsInstance(k.get_variables(), list)
        self.assertEqual(len(k.get_variables()), 2)

    def test_dof_feature_scale(self):
        k = FeatureScaled(MaternOneHalf)
        k = k(Mock2)

        self.assertEqual(k.dof, 2 + 2)

        k = FeatureScaled(MaternOneHalf)
        k = k(Mock5)

        self.assertEqual(k.dof, 2 + 5)

    def test_dof_feature_scale_complex(self):
        k = FeatureScaled(MaternOneHalf) + FeatureScaled(Linear)
        k = k(Mock2)

        self.assertEqual(k.dof, 2 + 2 + 2 + 3)
        self.assertEqual(len(k.kernel().trainable_variables), 1+1+2+3)

        k = FeatureScaled(FeatureScaled(MaternOneHalf) + Linear)
        k = k(Mock5)

        self.assertEqual(k.dof, 5 + 5 + 2 + 3)
        self.assertEqual(len(k.kernel().trainable_variables), 1+1+2+3)

    def test_number_of_variables(self):
        k = MaternOneHalf
        k = k(Mock5)
        self.assertEqual(len(k.kernel().trainable_variables), 2)

        k = Linear
        k = k(Mock5)
        self.assertEqual(len(k.kernel().trainable_variables), 3)

        k = Cubic
        k = k(Mock5)
        self.assertEqual(len(k.kernel().trainable_variables), 3)
        self.assertIsInstance(k.get_variables(), list)
        self.assertEqual(len(k.get_variables()), 3)

    def test_addition_dof_var(self):
        kc = Linear + Linear + MaternOneHalf
        ko = kc(Mock5)

        # variables
        self.assertEqual(len(ko.kernel().trainable_variables), 8)
        # dof
        self.assertEqual(ko.dof, 8)

    def test_multiplication_dof_var(self):
        kc = (Linear + Linear) * MaternOneHalf
        ko = kc(Mock5)

        # variables
        self.assertEqual(len(ko.kernel().trainable_variables), 8)
        # dof
        self.assertEqual(ko.dof, 8)
        self.assertIsInstance(ko.get_variables(), list)
        self.assertEqual(len(ko.get_variables()), 8)

    def test_expsinsquared(self):
        kc = ExpSinSquared(Mock2)
        self.assertEqual(len(kc.kernel().trainable_variables), 3)
        self.assertIsInstance(kc.get_variables(), list)
        self.assertEqual(len(kc.get_variables()), 3)

class TestsUID(unittest.TestCase):
    def testLinear(self):
        k = Linear
        self.assertEqual(k._uid, (('Linear',),))

    def test_LL(self):
        k = Linear + Linear
        self.assertEqual(k._uid, (('Linear',), ('Linear',)))

    def test_LM(self):
        k = MaternOneHalf + Linear
        self.assertEqual(k._uid, (('Linear',), ('MaternOneHalf',)))

        k = Linear + MaternOneHalf
        self.assertEqual(k._uid, (('Linear',), ('MaternOneHalf',)))

    def test_triple(self):
        k = Quadratic + MaternOneHalf + Linear
        self.assertEqual(k._uid, (('Linear',), ('MaternOneHalf',), ('Quadratic',)))

class TestsUID_mul(unittest.TestCase):
    def testLL(self):
        k = Linear * Linear
        self.assertEqual(k._uid, (('Linear', 'Linear'), ))

    def test_LM(self):
        k = MaternOneHalf * Linear
        self.assertEqual(k._uid, (('Linear', 'MaternOneHalf'),))

        k = Linear * MaternOneHalf
        self.assertEqual(k._uid, (('Linear', 'MaternOneHalf'),))

    def test_triple(self):
        k = Linear * MaternOneHalf * Quadratic
        self.assertEqual(k._uid, (('Linear', 'MaternOneHalf', 'Quadratic'),))

        k = Quadratic * Linear * MaternOneHalf
        self.assertEqual(k._uid, (('Linear', 'MaternOneHalf', 'Quadratic'),))

        k = Quadratic * MaternOneHalf * Linear
        self.assertEqual(k._uid, (('Linear', 'MaternOneHalf', 'Quadratic'),))

    def test_addition_distributive(self):
        k = (Quadratic + MaternOneHalf) * Linear
        self.assertEqual(k._uid, (
            ('Linear', 'MaternOneHalf'),
            ('Linear', 'Quadratic'),))

        k = (Quadratic + MaternOneHalf) * Linear + Linear
        self.assertEqual(k._uid, (
            ('Linear',),
            ('Linear', 'MaternOneHalf'),
            ('Linear', 'Quadratic'),
        ))

        k = (Quadratic + MaternOneHalf + Linear) * Linear
        self.assertEqual(k._uid, (
            ('Linear', 'Linear'),
            ('Linear', 'MaternOneHalf'),
            ('Linear', 'Quadratic'),
        ))

        k = (Quadratic + MaternOneHalf + Linear) * Linear * Cubic
        self.assertEqual(k._uid, (
            ('Cubic', 'Linear', 'Linear'),
            ('Cubic', 'Linear', 'MaternOneHalf'),
            ('Cubic', 'Linear', 'Quadratic'),
        ))

    def test_distributive_2(self):
        k = (Linear + Cubic) * (Quadratic + MaternOneHalf)
        self.assertEqual(k._uid, (
            ('Cubic', 'MaternOneHalf'),
            ('Cubic', 'Quadratic'),
            ('Linear', 'MaternOneHalf'),
            ('Linear', 'Quadratic'),
        ))

@unittest.skip('The output is in the question ...')
class TestsUID_functor(unittest.TestCase):
    def test_L(self):
        k = FeatureScaled(Linear)
        self.assertTrue(False)

class Tests_distance_jaccard(unittest.TestCase):
    def test_LL(self):
        a = Linear
        b = Linear
        self.assertAlmostEqual(kernel_similarity_measure_jaccard(a, b), 1.)

    def test_LM(self):
        a = Linear
        b = MaternOneHalf
        self.assertAlmostEqual(kernel_similarity_measure_jaccard(a, b), 0.)

    def test_L_LM_symetry(self):
        a = Linear
        b = Linear + MaternOneHalf
        self.assertAlmostEqual(kernel_similarity_measure_jaccard(a, b), 0.5)

        a = Linear + MaternOneHalf
        b = Linear
        self.assertAlmostEqual(kernel_similarity_measure_jaccard(a, b), 0.5)

    def test_L_LM_symetry(self):
        a = Linear * MaternOneHalf
        b = Linear + MaternOneHalf
        self.assertAlmostEqual(kernel_similarity_measure_jaccard(a, b), 0.0)

class Tests_distance_matching(unittest.TestCase):
    def test_LL(self):
        a = Linear
        b = Linear
        self.assertAlmostEqual(kernel_similarity_measure_best_matching(a, b), 1.)

    def test_LM(self):
        a = Linear
        b = MaternOneHalf
        self.assertAlmostEqual(kernel_similarity_measure_best_matching(a, b), 0.)

    def test_L_LM_symetry(self):
        a = Linear
        b = Linear + MaternOneHalf
        self.assertAlmostEqual(kernel_similarity_measure_best_matching(a, b), 0.5)

        a = Linear + MaternOneHalf
        b = Linear
        self.assertAlmostEqual(kernel_similarity_measure_best_matching(a, b), 0.5)

    def test_L_LM_symetry(self):
        a = Linear * MaternOneHalf
        b = Linear + MaternOneHalf
        self.assertAlmostEqual(kernel_similarity_measure_best_matching(a, b), 1./5.)

        a = Linear * MaternOneHalf + Linear * MaternOneHalf
        b = Linear + MaternOneHalf
        self.assertAlmostEqual(kernel_similarity_measure_best_matching(a, b), 1./3.)

        a = Linear * MaternOneHalf + Linear
        b = Linear + MaternOneHalf
        self.assertAlmostEqual(kernel_similarity_measure_best_matching(a, b), 3./5.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
