from typing import NamedTuple


class _MRVI_REGISTRY_KEYS_NT(NamedTuple):
    SAMPLE_KEY: str = "sample"
    CATEGORICAL_NUISANCE_KEYS: str = "categorical_nuisance_keys"


MRVI_REGISTRY_KEYS = _MRVI_REGISTRY_KEYS_NT()
