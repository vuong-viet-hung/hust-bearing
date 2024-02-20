import itertools
import random
import re
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from hust_bearing.data.module import BearingDataModule
from hust_bearing.data.dataset import BearingDataset


class CWRU(BearingDataModule):
    _classes: list[tuple[str, int]] = [("Normal", 0)]
    _classes.extend(itertools.product(["B", "IR", "OR"], [7, 14, 21]))
    _dir_name_regex = re.compile(
        r"""
        ([a-zA-Z]+)  # Fault
        (\d{3})?  # Size
        (?:@(\d+))?  # Location
        _
        (\d)  # Load
        """,
        re.VERBOSE,
    )
    _test_size = 500

    def setup(self, stage: str) -> None:
        paths = list(self._data_dir.glob("**/*.mat"))
        valid_paths = self._drop_invalid(paths)
        filtered_paths = self._filter_by_load(valid_paths)
        sampled_paths = self._sample(filtered_paths)

        targets = [self._target_from(path.parent.name) for path in sampled_paths]
        fit_paths, test_paths, fit_targets, test_targets = train_test_split(
            sampled_paths, targets, test_size=self._test_size, stratify=targets
        )

        if stage in {"fit", "validate"}:
            train_paths, val_paths, train_targets, val_targets = train_test_split(
                fit_paths, fit_targets, test_size=self._test_size, stratify=fit_targets
            )
            self._train_ds = BearingDataset(train_paths, train_targets)
            self._val_ds = BearingDataset(val_paths, val_targets)

        elif stage in {"test", "predict"}:
            self._test_ds = BearingDataset(test_paths, test_targets)

    def _drop_invalid(self, paths: list[Path]) -> list[Path]:
        classes = set(self._classes)
        return [path for path in paths if self._label_from(path.parent.name) in classes]

    def _filter_by_load(self, paths: list[Path]) -> list[Path]:
        if self._load is None:
            return paths
        return [
            path for path in paths if self._load_from(path.parent.name) == self._load
        ]

    def _sample(self, paths: list[Path]) -> list[Path]:
        if self._num_samples is None:
            return paths

        paths_grouped_by_label = self._group_paths_by_label(paths)

        use_balance_sampling = all(
            len(path_group) >= self._num_samples
            for path_group in paths_grouped_by_label.values()
        )

        if not use_balance_sampling:
            return random.sample(paths, self._num_samples)

        num_samples_per_class, remainder = divmod(self._num_samples, len(self._classes))
        sampled_paths: list[Path] = []
        for idx, path_group in enumerate(paths_grouped_by_label.values()):
            if idx < remainder:
                group_num_samples = num_samples_per_class + 1
            group_sampled_paths = random.sample(path_group, group_num_samples)
            sampled_paths.extend(group_sampled_paths)
        return sampled_paths

    def _group_paths_by_label(
        self, paths: list[Path]
    ) -> dict[tuple[str, int], list[Path]]:
        paths_grouped_by_target: dict[tuple[str, int], list[Path]] = defaultdict(list)
        for path in paths:
            label = self._label_from(path.parent.name)
            paths_grouped_by_target[label].append(path)
        return paths_grouped_by_target

    def _target_from(self, dir_name: str) -> int:
        label = self._label_from(dir_name)
        return self._classes.index(label)

    def _label_from(self, dir_name: str) -> tuple[str, int]:
        fault_type = self._fault_type_from(dir_name)
        fault_size = self._fault_size_from(dir_name)
        return fault_type, fault_size

    def _fault_type_from(self, dir_name: str) -> str:
        return self._parse(dir_name).group(1)

    def _fault_size_from(self, dir_name: str) -> int:
        fault_size = self._parse(dir_name).group(2)
        return int(fault_size) if fault_size is not None else 0

    def _load_from(self, dir_name: str) -> int:
        return int(self._parse(dir_name).group(4))

    def _parse(self, dir_name: str) -> re.Match[str]:
        match = self._dir_name_regex.fullmatch(dir_name)
        if match is None:
            raise ValueError
        return match
