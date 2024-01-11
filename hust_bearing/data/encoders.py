from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Encoder(ABC):
    @abstractmethod
    def encode_label(self, label: str) -> int:
        pass

    @abstractmethod
    def decode_label(self, label: int) -> str:
        pass

    def encode_labels(self, labels: npt.NDArray[np.str_]) -> npt.NDArray[np.int64]:
        return np.vectorize(self.encode_label)(labels)

    def decode_labels(self, labels: npt.NDArray[np.str_]) -> npt.NDArray[np.int64]:
        return np.vectorize(self.encode_label)(labels)


class CWRUEncoder(Encoder):
    _classes = ["Normal", "B", "IR", "OR"]

    def encode_label(self, label: str) -> int:
        return self._classes.index(label)

    def decode_label(self, label: int) -> str:
        return self._classes[label]


class HUSTEncoder(Encoder):
    _classes = ["N", "B", "I", "O", "IB", "IO", "OB"]

    def encode_label(self, label: str) -> int:
        return self._classes.index(label)

    def decode_label(self, label: int) -> str:
        return self._classes[label]
