from typing import ClassVar

BIPOP: RestartStrategy
COTN: CorrectionMethod
COUNT: CorrectionMethod
COVARIANCE: MatrixAdaptationType
CSA: StepSizeAdaptation
DEFAULT: RecombinationWeights
EQUAL: RecombinationWeights
GAUSSIAN: BaseSampler
HALF_POWER_LAMBDA: RecombinationWeights
HALTON: BaseSampler
IPOP: RestartStrategy
LPXNES: StepSizeAdaptation
MATRIX: MatrixAdaptationType
MIRROR: CorrectionMethod
MIRRORED: Mirror
MSR: StepSizeAdaptation
MXNES: StepSizeAdaptation
NONE: MatrixAdaptationType
SEPERABLE: MatrixAdaptationType
PAIRWISE: Mirror
PSR: StepSizeAdaptation
RESTART: RestartStrategy
SATURATE: CorrectionMethod
SOBOL: BaseSampler
STOP: RestartStrategy
TOROIDAL: CorrectionMethod
TPA: StepSizeAdaptation
UNIFORM_RESAMPLE: CorrectionMethod
XNES: StepSizeAdaptation

class BaseSampler:
    __members__: ClassVar[dict] = ...  # read-only
    GAUSSIAN: ClassVar[BaseSampler] = ...
    HALTON: ClassVar[BaseSampler] = ...
    SOBOL: ClassVar[BaseSampler] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CorrectionMethod:
    __members__: ClassVar[dict] = ...  # read-only
    COTN: ClassVar[CorrectionMethod] = ...
    COUNT: ClassVar[CorrectionMethod] = ...
    MIRROR: ClassVar[CorrectionMethod] = ...
    NONE: ClassVar[CorrectionMethod] = ...
    SATURATE: ClassVar[CorrectionMethod] = ...
    TOROIDAL: ClassVar[CorrectionMethod] = ...
    UNIFORM_RESAMPLE: ClassVar[CorrectionMethod] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class MatrixAdaptationType:
    __members__: ClassVar[dict] = ...  # read-only
    COVARIANCE: ClassVar[MatrixAdaptationType] = ...
    MATRIX: ClassVar[MatrixAdaptationType] = ...
    SEPERABLE: ClassVar[MatrixAdaptationType] = ...
    NONE: ClassVar[MatrixAdaptationType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Mirror:
    __members__: ClassVar[dict] = ...  # read-only
    MIRRORED: ClassVar[Mirror] = ...
    NONE: ClassVar[Mirror] = ...
    PAIRWISE: ClassVar[Mirror] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class RecombinationWeights:
    __members__: ClassVar[dict] = ...  # read-only
    DEFAULT: ClassVar[RecombinationWeights] = ...
    EQUAL: ClassVar[RecombinationWeights] = ...
    HALF_POWER_LAMBDA: ClassVar[RecombinationWeights] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class RestartStrategy:
    __members__: ClassVar[dict] = ...  # read-only
    BIPOP: ClassVar[RestartStrategy] = ...
    IPOP: ClassVar[RestartStrategy] = ...
    NONE: ClassVar[RestartStrategy] = ...
    RESTART: ClassVar[RestartStrategy] = ...
    STOP: ClassVar[RestartStrategy] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class StepSizeAdaptation:
    __members__: ClassVar[dict] = ...  # read-only
    CSA: ClassVar[StepSizeAdaptation] = ...
    LPXNES: ClassVar[StepSizeAdaptation] = ...
    MSR: ClassVar[StepSizeAdaptation] = ...
    MXNES: ClassVar[StepSizeAdaptation] = ...
    PSR: ClassVar[StepSizeAdaptation] = ...
    TPA: ClassVar[StepSizeAdaptation] = ...
    XNES: ClassVar[StepSizeAdaptation] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...
