import numpy as np
from scipy.stats import norm
from sobol_seq import i4_sobol
from ghalton import Halton


def gaussian_sampling(d):
    while True:
        yield np.random.randn(d, 1)


def sobol_sampling(d):
    seed = np.random.randint(2, d**2)
    while True:
        sample, seed = i4_sobol(d, max(seed, 2))
        yield norm.ppf(sample).reshape(-1, 1)


def halton_sampling(d):
    halton = Halton(d)
    while True:
        yield norm.ppf(halton.get(1)[0]).reshape(-1, 1)


def mirrored_sampling(b):
    for sample in b:
        yield sample
        yield sample * -1


def orthogonal_sampling(b, n_samples):
    # TODO: ask Hao
    samples = []
    for sample in b:
        if len(samples) == n_samples:
            samples = np.hstack(samples).T
            L = np.linalg.norm(samples, axis=1).reshape(n_samples, -1)
            Q, _ = np.linalg.qr(samples.reshape(
                max(samples.shape), min(samples.shape)))
            if n_samples < max(samples.shape):
                samples = (Q.T * L).tolist()
            else:
                samples = (Q * L).tolist()
            while any(samples):
                yield np.array(samples.pop()).reshape(-1, 1)
        else:
            samples.append(sample)


########## OLD CODE ############################
class GaussianSampling:
    def __init__(self, d: int):
        self.d = d

    def __next__(self) -> np.ndarray:
        return np.random.randn(*(self.d, 1))


class QuasiGaussianSobolSampling:
    """
        A quasi-Gaussian sampler based on a Sobol sequence

        :param n:       Dimensionality of the vectors to be sampled
        :param shape:   String to select between whether column (``'col'``) or row (``'row'``) vectors should be
                        returned. Defaults to column vectors
    """

    def __init__(self, n, seed=None):
        self.n = n
        self.shape = (n, 1)
        if seed is None or seed < 2:
            # seed=1 will give a null-vector as first result
            self.seed = np.random.randint(2, n**2)
        else:
            self.seed = seed

    def __next__(self):
        """
        Draw the next sample from the Sampler

        :return:    A new vector sampled from a Sobol sequence with mean 0 and standard deviation 1
        """
        vec, seed = i4_sobol(self.n, self.seed)
        self.seed = seed if seed > 1 else 2
        return np.array(norm.ppf(vec)).reshape(self.shape)


class QuasiGaussianHaltonSampling:
    """
        A quasi-Gaussian sampler based on a Halton sequence

        :param n:       Dimensionality of the vectors to be sampled
        :param shape:   String to select between whether column (``'col'``) or row (``'row'``) vectors should be
                        returned. Defaults to column vectors
    """

    def __init__(self, n):
        self.n = n
        self.shape = (n, 1)
        self.halton = Halton(n)

    def __next__(self):
        """
            Draw the next sample from the Sampler

            :return:    A new vector sampled from a Halton sequence with mean 0 and standard deviation 1
        """
        vec = self.halton.get(1)[0]
        return norm.ppf(vec).reshape(self.shape)


class MirroredSampling:
    """
    A sampler to create mirrored samples using some base sampler (Gaussian by default)
    Returns a single vector each time, while remembering the internal state of whether the ``next()`` should return
    a new sample, or the mirror of the previous one.

    :param d:               Dimensionality of the vectors to be sampled
    :param base_sampler:    A different Sampling object from which samples to be mirrored are drawn. If no
                            base_sampler is given, a :class:`~GaussianSampling` object will be
                            created and used.
    """

    def __init__(self, base_sampler):
        self.mirror_next = False
        self.last_sample = None
        self.base_sampler = base_sampler

    def __next__(self):
        """
        Draw the next sample from the Sampler
        :return:    A new vector, alternating between a new sample from
                    the base_sampler and a mirror of the last.
        """
        if self.mirror_next:
            sample = self.last_sample * -1
        else:
            sample = next(self.base_sampler)
            self.last_sample = sample

        self.mirror_next = not self.mirror_next
        return sample


class OrthogonalSampling:
    """
    A sampler to create orthogonal samples using some base sampler (Gaussian as default)

    :param n:               Dimensionality of the vectors to be sampled
    :param lambda_:         Number of samples to be drawn and orthonormalized per generation
                            returned. Defaults to column vectors
    :param base_sampler:    A different Sampling object from which samples to be mirrored are drawn. If no
                            base_sampler is given, a :class:`~GaussianSampling` object will be
                            created and used.
    """

    def __init__(self, d, lambda_, base_sampler):
        self.d = d
        self.base_sampler = base_sampler

        self.num_samples = int(lambda_)
        self.current_sample = 0
        self.samples = None

    def __next__(self):
        """
        Draw the next sample from the Sampler
        :return:    A new vector sampled from a set of orthonormalized vectors,
                    originally drawn from base_sampler
        """
        if self.current_sample % self.num_samples == 0:
            self.current_sample = 0
            invalid_samples = True
            while invalid_samples:
                invalid_samples = self.__generate_samples()

        self.current_sample += 1
        return self.samples[self.current_sample - 1]

    def __generate_samples(self):
        """ Draw <num_samples> new samples from the base_sampler,
            orthonormalize them and store to be drawn from """
        samples = []
        lengths = []
        for i in range(self.num_samples):
            sample = next(self.base_sampler)
            samples.append(sample)
            lengths.append(np.linalg.norm(sample))

        num_samples = self.num_samples if self.num_samples <= self.d else self.d
        samples[:num_samples] = self.__gram_schmidt(samples[:num_samples])

        for i in range(num_samples):
            samples[i] *= lengths[i]

        self.samples = samples
        return np.any(np.isnan(samples))

    def __gram_schmidt(self, vectors):
        """ Implementation of the Gram-Schmidt process for
        orthonormalizing a set of vectors """
        num_vectors = len(vectors)
        lengths = np.zeros(num_vectors)
        lengths[0] = np.linalg.norm(vectors[0])

        for i in range(1, num_vectors):
            for j in range(i):
                vec_i = vectors[i]
                vec_j = vectors[j]
                # This will prevent Runtimewarning (Division over zero)
                if lengths[j]:
                    vectors[i] = vec_i - vec_j * \
                        (np.dot(vec_i.T, vec_j) / lengths[j] ** 2)
            lengths[i] = np.linalg.norm(vectors[i])

        for i, vec in enumerate(vectors):
            # In the rare, but not uncommon cases of this producing 0-vectors, we simply replace it with a random one
            if lengths[i] == 0:
                new_vector = next(self.base_sampler)
                vectors[i] = new_vector / np.linalg.norm(new_vector)
            else:
                vectors[i] = vec / lengths[i]

        return vectors


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum([np.dot(v, b) * b for b in basis])
        # if (w > 1e-10).any():
        basis.append(w / np.linalg.norm(w))
    return np.array(basis)


def gs(X, norm=True):
    Y = X[0:1, :].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag(
            (X[i, :].dot(Y.T) / np.linalg.norm(Y, axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i, :] - proj.sum(0)))

    return np.diag(1 / np.linalg.norm(Y, axis=1)).dot(Y)
