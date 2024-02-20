import random
import re
from pathlib import Path

from sklearn.model_selection import train_test_split

from hust_bearing.data.module import BearingDataModule
from hust_bearing.data.dataset import BearingDataset


class CWRU(BearingDataModule):
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
    _classes = ["Normal", "B", "IR", "OR"]

    def setup(self, stage: str) -> None:
        paths = list(self._data_dir.glob("**/*.mat"))
        filtered_paths = self._filter_by_load(paths)
        sampled_paths = self._sample_paths(filtered_paths)
        sampled_targets = [
            self._target_from(path.parent.name) for path in sampled_paths
        ]
        fit_paths, test_paths, fit_targets, test_targets = train_test_split(
            sampled_paths, sampled_targets, test_size=800, stratify=sampled_targets
        )

        if stage in {"fit", "validate"}:
            train_paths, val_paths, train_targets, val_targets = train_test_split(
                fit_paths, fit_targets, test_size=200, stratify=fit_targets
            )
            self._train_ds = BearingDataset(train_paths, train_targets)
            self._val_ds = BearingDataset(val_paths, val_targets)

        elif stage in {"test", "predict"}:
            self._test_ds = BearingDataset(test_paths, test_targets)

    def _filter_by_load(self, paths: list[Path]) -> list[Path]:
        if self._load is None:
            return paths
        return [
            path for path in paths if self._load_from(path.parent.name) == self._load
        ]

    def _sample_paths(self, paths: list[Path]) -> list[Path]:
        if self._num_samples is None:
            return paths

        paths_grouped_by_target: list[list[Path]] = [[] for _ in self._classes]
        for path in paths:
            target = self._target_from(path.parent.name)
            paths_grouped_by_target[target].append(path)

        use_balance_sampling = all(
            len(path_group) >= self._num_samples
            for path_group in paths_grouped_by_target
        )

        if not use_balance_sampling:
            return random.sample(paths, self._num_samples)

        sampled_paths = []
        num_samples_per_class, remainder = divmod(self._num_samples, len(self._classes))
        for idx, path_group in enumerate(paths_grouped_by_target):
            if idx < remainder:
                num_samples_per_class += 1
            sampled_paths.extend(random.sample(path_group, num_samples_per_class))
        return sampled_paths

    def _target_from(self, dir_name: str) -> int:
        return self._classes.index(self._parse(dir_name).group(1))

    def _load_from(self, dir_name: str) -> int:
        return int(self._parse(dir_name).group(4))

    def _parse(self, dir_name: str) -> re.Match[str]:
        match = self._dir_name_regex.fullmatch(dir_name)
        if match is None:
            raise ValueError
        return match
