import enum


class RecombinationWeights(enum.Enum):
    DEFAULT = 0

    EQUAL = 1

    EXPONENTIAL = 2

class BaseSampler(enum.Enum):
    UNIFORM = 0

    SOBOL = 1

    HALTON = 2

class SampleTranformerType(enum.Enum):
    NONE = 0

    GAUSSIAN = 1

    SCALED_UNIFORM = 2

    LAPLACE = 3

    LOGISTIC = 4

    CAUCHY = 5

    DOUBLE_WEIBULL = 6

class Mirror(enum.Enum):
    NONE = 0

    MIRRORED = 1

    PAIRWISE = 2

class StepSizeAdaptation(enum.Enum):
    CSA = 0

    TPA = 1

    MSR = 2

    XNES = 3

    MXNES = 4

    LPXNES = 5

    PSR = 6

    SR = 7

    SA = 8

class CorrectionMethod(enum.Enum):
    NONE = 0

    MIRROR = 1

    COTN = 2

    UNIFORM_RESAMPLE = 3

    SATURATE = 4

    TOROIDAL = 5

    RESAMPLE = 6

class RestartStrategy(enum.Enum):
    NONE = 0

    STOP = 2

    RESTART = 1

    IPOP = 3

    BIPOP = 4

class MatrixAdaptationType(enum.Enum):
    COVARIANCE = 1

    NONE = 0

    MATRIX = 2

    SEPARABLE = 3

    CHOLESKY = 4

    CMSA = 5

    COVARIANCE_NO_EIGV = 6

    NATURAL_GRADIENT = 7

class CenterPlacement(enum.Enum):
    X0 = 0

    ZERO = 1

    UNIFORM = 2

    CENTER = 3

    NOVELTY_WEIGHTED = 4

    MAXIMIN_TABOO = 5

class RepellingType(enum.Enum):
    NONE = 0

    COVERAGE = 1

    ADAPTIVE = 2

class RepellingMetric(enum.Enum):
    EUCLIDEAN = 0

    CURRENT = 1

    ARCHIVED = 2
