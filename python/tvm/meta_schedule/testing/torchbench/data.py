# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Benchmark Data Storage Interface
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import pickle
import shutil
from contextlib import nullcontext
from typing import ContextManager, List, Tuple

import torch

from tvm import meta_schedule as ms

from .utils import DisallowedOperator


@dataclass
class TaskExtractionConfig:
    model: str
    target: tvm.target.Target
    cast_to_float32: bool
    input_batch_size: int

    disallowed_ops: List[DisallowedOperator]


@dataclass
class TaskExtractionResult:
    pass


@dataclass
class TuningConfig:
    model: str
    target: tvm.target.Target

    strategy: str
    max_trials_global: int
    max_trials_per_task: int

    adaptive_training: bool

    runner_evaluator_config: EvaluatorConfig
    rpc_config: Optional[ms.runner.RPCConfig]


@dataclass
class TuningResult:
    pass


@dataclass
class EvalConfig:
    model: str
    target: tvm.target.Target
    cast_to_float32: bool
    input_batch_size: int

    tvm_backend: str

    result_comparison_metric: ResultComparisonMetric

    warmup_rounds: int
    repeat: int


@dataclass
class EvalResult:
    pass


class ExtractedTasksStorage(ABC):
    @abstractmethod
    def get_extracted_tasks(self) -> List[ms.ExtractedTask]:
        raise NotImplementedError()

    @abstractmethod
    def set_extracted_tasks(self, tasks: List[ms.ExtractedTask]):
        raise NotImplementedError()

    @abstractmethod
    def get_subgraph(self, graph_id: int) -> torch.fx.GraphModule:
        raise NotImplementedError()

    @abstractmethod
    def set_subgraph(self, graph_id: int, graph_module: torch.fx.GraphModule):
        raise NotImplementedError()

    @abstractmethod
    def get_subgraph_example_inputs(self, graph_id: int) -> Tuple[torch.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def set_subgraph_example_inputs(self, graph_id: int, example_inputs: Tuple[torch.Tensor]):
        raise NotImplementedError()


class TuningDataStorage(ABC):
    @abstractmethod
    def with_metaschedule_logging_dir(self) -> ContextManager[str]:
        raise NotImplementedError()

    @abstractmethod
    def get_metaschedule_database(self) -> ms.Database:
        raise NotImplementedError()

    @abstractmethod
    def set_metaschedule_database(self, database: ms.Database):
        raise NotImplementedError()


class BenchmarkDataStorage(ABC):
    @abstractmethod
    def get_model_expected_output(self):
        raise NotImplementedError()

    @abstractmethod
    def set_model_expected_output(self, expected):
        raise NotImplementedError()

    @abstractmethod
    def get_model_actual_output(self):
        raise NotImplementedError()

    @abstractmethod
    def set_model_actual_output(self, actual):
        raise NotImplementedError()


class LocalExtractedTasksStorage(ExtractedTasksStorage):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(os.path.join(root_dir, "subgraphs"), exist_ok=True)

    @property
    def extracted_tasks_path(self) -> str:
        return os.path.join(self.root_dir, "extracted_tasks")

    def get_extracted_tasks(self) -> List[ms.ExtractedTask]:
        with open(self.extracted_tasks_path, "rb") as f:
            return pickle.load(f)

    def set_extracted_tasks(self, tasks: List[ms.ExtractedTask]):
        with open(self.extracted_tasks_path, "wb") as f:
            pickle.dump(tasks, f)

    def _get_subgraph_path(self, graph_id: str) -> str:
        return os.path.join(self.root_dir, "subgraphs", f"graph_module_{graph_id}")

    def get_subgraph(self, graph_id: str) -> torch.fx.GraphModule:
        return torch.load(self._get_subgraph_path(graph_id))

    def set_subgraph(self, graph_id: str, graph_module: torch.fx.GraphModule):
        torch.save(graph_module, self._get_subgraph_path(graph_id))

    def _get_subgraph_example_inputs_path(self, graph_id: str) -> str:
        return os.path.join(self.root_dir, "subgraphs", f"example_inputs_{graph_id}")

    def get_subgraph_example_inputs(self, graph_id: str) -> Tuple[torch.Tensor]:
        return torch.load(self._get_subgraph_example_inputs_path(graph_id))

    def set_subgraph_example_inputs(self, graph_id: str, example_inputs: Tuple[torch.Tensor]):
        torch.save(example_inputs, self._get_subgraph_example_inputs_path(graph_id))


class LocalTuningDataStorage(TuningDataStorage):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(os.path.join(root_dir, "meta_schedule"), exist_ok=True)

    @property
    def _ms_workdir(self) -> str:
        return os.path.join(self.root_dir, "meta_schedule")

    @property
    def _workload_json_path(self) -> str:
        return os.path.join(self._ms_workdir, "database_workload.json")

    @property
    def _tuning_record_json_path(self) -> str:
        return os.path.join(self._ms_workdir, "database_tuning_record.json")

    def with_metaschedule_logging_dir(self) -> ContextManager[str]:
        return nullcontext(self._ms_workdir)

    def get_metaschedule_database(self) -> ms.Database:
        return ms.database.JSONDatabase(
            path_workload=self._workload_json_path,
            path_tuning_record=self._tuning_record_json_path,
        )

    def set_metaschedule_database(self, database: ms.Database):
        if isinstance(database, ms.database.database.JSONDatabase):
            if database.path_workload != self._workload_json_path:
                shutil.copy(database.path_workload, self._workload_json_path)
            if database.path_tuning_record != self._tuning_record_json_path:
                shutil.copy(database.path_tuning_record, self._tuning_record_json_path)
        else:
            raise RuntimeError(f"Cannot save MetaSchedule database with type {type(database)}")


class LocalBenchmarkDataStorage(BenchmarkDataStorage):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def get_model_expected_output(self):
        return torch.load(os.path.join(self.root_dir, "expected_output.pt"))

    def set_model_expected_output(self, expected):
        torch.save(expected, os.path.join(self.root_dir, "expected_output.pt"))

    def get_model_actual_output(self):
        return torch.load(os.path.join(self.root_dir, "actual_output.pt"))

    def set_model_actual_output(self, actual):
        torch.save(actual, os.path.join(self.root_dir, "actual_output.pt"))
