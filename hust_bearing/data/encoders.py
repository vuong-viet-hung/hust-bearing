from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Encoder(ABC):
    @abstractmethod
    def encode_labels(self, labels: npt.NDArray[np.str_]) -> npt.NDArray[np.int64]:
        pass

    @abstractmethod
    def decode_targets(self, targets: npt.NDArray[np.str_]) -> npt.NDArray[np.int64]:
        pass


class CWRUEncoder(Encoder):
    _classes = np.array(["Normal", "B", "IR", "OR"])

    def encode_labels(self, labels: npt.NDArray[np.object_]) -> npt.NDArray[np.int64]:
        return np.argmax(np.expand_dims(self._classes, axis=1) == labels, axis=0)

    def decode_targets(self, targets: npt.NDArray[np.int64]) -> npt.NDArray[np.str_]:
        return self._classes[targets]


class HUSTEncoder(Encoder):
    _classes = np.array(["N", "B", "I", "O", "IB", "IO", "OB"])

    def encode_labels(self, labels: npt.NDArray[np.object_]) -> npt.NDArray[np.int64]:
        return np.argmax(np.expand_dims(self._classes, axis=1) == labels, axis=0)

    def decode_targets(self, targets: npt.NDArray[np.int64]) -> npt.NDArray[np.str_]:
        return self._classes[targets]
