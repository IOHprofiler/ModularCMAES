import numpy as np


class GaussianSampling:
    def __init__(self, d: int):
        self.d = d

    def next(self) -> np.ndarray:
        return np.random.randn(*(self.d, 1))


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

    def next(self):
        """
        Draw the next sample from the Sampler
        :return:    A new vector, alternating between a new sample from 
                    the base_sampler and a mirror of the last.
        """
        if self.mirror_next:
            sample = self.last_sample * -1
        else:
            sample = self.base_sampler.next()
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

    def next(self):
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
            sample = self.base_sampler.next()
            samples.append(sample)
            lengths.append(np.linalg.norm(sample))

        num_samples = self.num_samples if self.num_samples <= self.d else self.d
        samples[:num_samples] = self.__gram_schmidt(samples[:num_samples])
        for i in range(num_samples):
            samples[i] *= lengths[i]

        self.samples = samples
        # Are all generated samples np.any good? I.e. is there no 'nan' value anywhere?
        return np.any(np.isnan(samples))

    def __gram_schmidt(self, vectors):
        """ Implementation of the Gram-Schmidt process for orthonormalizing a set of vectors """
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
                new_vector = self.base_sampler.next()
                vectors[i] = new_vector / np.linalg.norm(new_vector)
            else:
                vectors[i] = vec / lengths[i]

        return vectors

    def reset(self):
        """
            Reset the internal state of this sampler, so the next sample is forced to be taken new.
        """
        self.current_sample = 0
        self.samples = None
        self.base_sampler.reset()
