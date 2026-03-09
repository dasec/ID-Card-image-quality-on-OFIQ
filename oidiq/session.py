from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Generic, overload

import numpy as np
from .utils import OIDIQConfig, QualityMetric, QualityMetricConfig, batching, config, deep_copy_dict
from .utils import creates as creates_alias
from PIL import Image
import time

import inspect


class PreProcessors:
    CROPPED_IMAGE = "cropped_image"
    ID_CARD_CORNERS = "id_card_corners"
    NORMALIZED_IMAGE = "normalized_image"
    NORMALIZED_FACE_MASK = "normalized_face_mask"
    NORMALIZED_FACE_BOXES = "normalized_face_boxes"
    NORMALIZED_FACE_BACKGROUND_MASK = "normalized_face_background_mask"
    NORMALIZED_FOREGROUND_MASK = "normalized_foreground_mask"
    NORMALIZED_LUMINANCE = "normalized_luminance"
    NORMALIZED_LUMINANCE_HISTOGRAM = "normalized_luminance_histogram"


Q = TypeVar("Q")
P = TypeVar("P")


class OIDIQBaseSession(ABC, Generic[Q, P]):
    def __init__(
        self,
        img: P,
        preprocessors: List["OIDIQPreprocessor"] | Dict[str, "OIDIQPreprocessor"] | None = None,
        metric_creators: List["OIDIQMetricCreator"] | Dict[str, "OIDIQMetricCreator"] | None = None,
        verbose: bool = False,
    ):
        self.raw_img = img
        # self.preprocessors = preprocessors
        # self.metric_creators = metric_creators
        self.preprocessors: Dict[str, "OIDIQPreprocessor"] = {}
        self.metric_creators: Dict[str, "OIDIQMetricCreator"] = {}
        self.verbose = verbose
        if isinstance(preprocessors, dict):
            self.preprocessors = preprocessors
        elif isinstance(preprocessors, list):
            for preprocessor in preprocessors:
                self.register_preprocessor(preprocessor)

        if isinstance(metric_creators, dict):
            self.metric_creators = metric_creators
        elif isinstance(metric_creators, list):
            for metric_creator in metric_creators:
                self.register_metric_creator(metric_creator)

    def get_scores(self, *names: str) -> List[Q]:
        r = []
        for name in names:
            r.append(self.get_score(name))
        return r

    @abstractmethod
    def _is_cached(self, name: str) -> bool:
        pass

    @abstractmethod
    def _get_cache(self, name: str) -> Any:
        pass

    @abstractmethod
    def _set_cache(self, name: str, value: Any) -> None:
        pass

    @abstractmethod
    def _delete_cached(self, name: str) -> None:
        pass

    @abstractmethod
    def reset_cache(self) -> None:
        pass

    @abstractmethod
    def _get_all_cached_keys(self) -> Iterable[str]:
        pass

    @abstractmethod
    def _is_save_requests(self) -> bool:
        pass

    @abstractmethod
    def _set_save_requests(self, value: bool) -> None:
        pass

    @abstractmethod
    def _get_requests(self) -> List[str]:
        pass

    def get_score(self, name: str) -> Q:
        if self._is_cached("score__" + name):
            return self._get_cache("score__" + name)
        if not name in self.metric_creators:
            raise KeyError(f"Metric '{name}' not found in session.")
        metric = self.metric_creators[name]
        if self.verbose:
            start_time = time.time()
        metric_result = metric.get(self, name)
        r = metric_result[name]
        if self.verbose:
            end_time = time.time()
            self.log(
                metric,
                f"Took {end_time - start_time:.4f} seconds for '{name}' and created keys: {list(metric_result.keys())}",
            )
        for k, v in metric_result.items():
            if k in self.metric_creators and self.metric_creators[k] is metric:
                self._set_cache("score__" + k, v)
        return r  # type: ignore

    def get_all_scores(self) -> Dict[str, Q]:
        all_r = {}
        for creates, metric in self.metric_creators.items():
            if creates in all_r:
                continue
            if self._is_cached("score__" + creates):
                all_r[creates] = self._get_cache("score__" + creates)
                continue
            if self.verbose:
                start_time = time.time()

            r = metric.get(self)

            if self.verbose:
                end_time = time.time()
                self.log(
                    metric,
                    f"Took {end_time - start_time:.4f} seconds for '{creates}' and created keys: {list(r.keys())}",
                )

            for k, v in r.items():
                if k in self.metric_creators and self.metric_creators[k] is metric:
                    self._set_cache("score__" + k, v)
                    all_r[k] = v
        return all_r

    def get_preprocessed_image(self, name: str) -> P:
        if self._is_cached(name):
            return self._get_cache(name)
        if name == "raw_image":
            return self.raw_img
        if not name in self.preprocessors:
            raise KeyError(f"Preprocessor for '{name}' not found in session.")
        preprocessor = self.preprocessors[name]
        if self.verbose:
            start_time = time.time()
        preprocess_result = preprocessor.get(self, name)
        if self.verbose:
            end_time = time.time()
            self.log(
                preprocessor,
                f"Took {end_time - start_time:.4f} seconds for '{name}' and created keys: {list(preprocess_result.keys())}",
            )
        for k, v in preprocess_result.items():
            if k in self.preprocessors and self.preprocessors[k] is preprocessor:
                self._set_cache(k, v)
        return self._get_cache(name)

    def register_preprocessor(self, preprocessor: "OIDIQPreprocessor", *use: str):
        if use:
            for name in use:
                self.preprocessors[name] = preprocessor
        else:
            for name in preprocessor.creates():
                self.preprocessors[name] = preprocessor

    def register_metric_creator(self, metric_creator: "OIDIQMetricCreator", *use: str):
        if use:
            for name in use:
                self.metric_creators[name] = metric_creator
        else:
            for name in metric_creator.creates():
                self.metric_creators[name] = metric_creator

    def registered_preprocessors(self) -> List[str]:
        return list(self.preprocessors.keys())

    def registered_metric_creators(self) -> List[str]:
        return list(self.metric_creators.keys())

    def get_raw_image(self) -> P:
        if self._is_save_requests():
            self._get_requests().append("raw_image")
        return self.raw_img

    def get_cropped_image(self) -> P:
        return self.get_preprocessed_image(PreProcessors.CROPPED_IMAGE)

    def get_id_card_corners(self) -> P:
        return self.get_preprocessed_image(PreProcessors.ID_CARD_CORNERS)
    

    def get_normalized_image(self) -> P:
        return self.get_preprocessed_image(PreProcessors.NORMALIZED_IMAGE)

    def get_normalized_face_mask(self) -> P:
        return self.get_preprocessed_image(PreProcessors.NORMALIZED_FACE_MASK)

    def get_normalized_face_boxes(self) -> P:
        return self.get_preprocessed_image(PreProcessors.NORMALIZED_FACE_BOXES)

    def get_normalized_face_background_mask(self) -> P:
        return self.get_preprocessed_image(PreProcessors.NORMALIZED_FACE_BACKGROUND_MASK)

    def get_normalized_foreground_mask(self) -> P:
        return self.get_preprocessed_image(PreProcessors.NORMALIZED_FOREGROUND_MASK)

    def get_normalized_luminance(self) -> P:
        return self.get_preprocessed_image(PreProcessors.NORMALIZED_LUMINANCE)

    def get_normalized_luminance_histogram(self) -> P:
        return self.get_preprocessed_image(PreProcessors.NORMALIZED_LUMINANCE_HISTOGRAM)

    def update_preprocessed_image(
        self, name: str, value: Any, graph: Dict[str, Tuple[Set[str], Set[str]]] | None = None
    ) -> Tuple[Set[str], Set[str]]:
        return self._update_value(name, value, delete=False, graph=graph)

    def delete_preprocessed_image(
        self, name: str, graph: Dict[str, Tuple[Set[str], Set[str]]] | None = None
    ) -> Tuple[Set[str], Set[str]]:
        return self._update_value(name, None, delete=True, graph=graph)

    def update_metric(
        self, name: str, value: QualityMetric, graph: Dict[str, Tuple[Set[str], Set[str]]] | None = None
    ) -> Tuple[Set[str], Set[str]]:
        return self._update_value("score__" + name, value, delete=False, graph=graph)

    def delete_metric(
        self, name: str, graph: Dict[str, Tuple[Set[str], Set[str]]] | None = None
    ) -> Tuple[Set[str], Set[str]]:
        return self._update_value("score__" + name, None, delete=True, graph=graph)

    def _update_value(
        self, name: str, value: Any, delete: bool, graph: Dict[str, Tuple[Set[str], Set[str]]] | None = None
    ) -> Tuple[Set[str], Set[str]]:
        if graph is None:
            if delete:
                if self._is_cached(name):
                    self._delete_cached(name)
                return set(), set()
            else:
                self._set_cache(name, value)
                return set(), set()

        queue = [name]
        deleted_scores = set()
        deleted_preprocess = set()
        while queue:
            current = queue.pop(0)
            if self._is_cached(current):
                if current.startswith("score__"):
                    deleted_scores.add(current[7:])
                else:
                    deleted_preprocess.add(current)
                self._delete_cached(current)
            for dependent in graph[current][1]:
                queue.append(dependent)

        if not delete:
            self._set_cache(name, value)

        self.log(
            self,
            f"{'Deleted' if delete else 'Updated'} '{name}'. Also affected scores: {deleted_scores}, preprocessors: {deleted_preprocess}",
        )
        return deleted_scores, deleted_preprocess

    def _list_requirements(self, name: str) -> List[str]:
        self._set_save_requests(True)
        self._get_requests().clear()
        if name.startswith("score__"):
            self.get_score(name[7:])
        else:
            self.get_preprocessed_image(name)
        self._set_save_requests(False)
        self._get_requests().pop(0)  # Remove the root request
        return list(self._get_requests())

    def _build_dependency_tree(self) -> Dict[str, Dict]:
        base_reqs = set()
        for metric in self.metric_creators.values():
            for name in metric.creates():
                base_reqs.add("score__" + name)
        for preprocessor in self.preprocessors.values():
            for name in preprocessor.creates():
                base_reqs.add(name)
        base_reqs.add("raw_image")
        all_requirements = {}

        for req in set(base_reqs):
            all_requirements[req] = self._list_requirements(req)
        all_requirements_refs = {}

        for k in all_requirements.keys():
            all_requirements_refs[k] = {}

        resolved = set()
        layers = []

        while len(resolved) < len(all_requirements):
            layers.append(set())
            for k, reqs in all_requirements.items():
                if k in resolved:
                    continue
                if len(reqs) == 0 or all(r in resolved for r in reqs):
                    reqs_copy = reqs.copy()
                    for layer_idx in range(len(layers) - 2, -1, -1):
                        for last_r in layers[layer_idx]:
                            if last_r in reqs_copy:
                                for sub_r in all_requirements[last_r]:
                                    reqs_copy.remove(sub_r)

                    for r in reqs_copy:
                        all_requirements_refs[k][r] = all_requirements_refs[r]
                    layers[-1].add(k)

            resolved.update(layers[-1])

        return all_requirements_refs

    def dependency_graph(self, tree: Dict | None = None) -> Dict[str, Tuple[Set[str], Set[str]]]:
        tree = tree or self._build_dependency_tree()
        graph: Dict[str, Tuple[Set[str], Set[str]]] = {}
        for k in tree.keys():
            graph[k] = (set(), set())
        for k, v in tree.items():
            for sub_k in v.keys():
                graph[k][0].add(sub_k)
                graph[sub_k][1].add(k)
        return graph

    def log(self, obj: "OIDIQBaseSession | OIDIQBaseExecutor | type", text: str):
        if self.verbose:
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            name = obj.__name__ if isinstance(obj, type) else obj.__class__.__name__
            print(f"[{t}] [{name}]: {text}")


class OIDIQBaseBatchSession(OIDIQBaseSession[List[QualityMetric], List[np.ndarray]]):
    @overload
    def __getitem__(self, idx: int) -> "OIDIQBaseSingleSession": ...
    @overload
    def __getitem__(self, idx: slice) -> "OIDIQBaseBatchSession": ...

    @abstractmethod
    def __getitem__(self, idx: int | slice) -> "OIDIQBaseBatchSession | OIDIQBaseSingleSession":
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_all_scores_transposed(self) -> List[Dict[str, QualityMetric]]:
        all_scores = self.get_all_scores()
        results = []
        for i in range(len(self)):
            result = {}
            for k, v in all_scores.items():
                result[k] = v[i]
            results.append(result)
        return results

    def get_scores_transposed(self, *names: str) -> List[List[QualityMetric]]:
        all_scores = self.get_scores(*names)
        results = []
        for i in range(len(self)):
            result = []
            for score_list in all_scores:
                result.append(score_list[i])
            results.append(result)
        return results


class OIDIQBaseSingleSession(OIDIQBaseSession[QualityMetric, np.ndarray]):
    @abstractmethod
    def to_batch(self) -> OIDIQBaseBatchSession:
        pass

    def __iter__(self):
        yield self


class OIDIQBatchSession(OIDIQBaseBatchSession):
    def __init__(
        self,
        imgs: Iterable[np.ndarray | str],
        preprocessors: List["OIDIQPreprocessor"] | Dict[str, "OIDIQPreprocessor"] | None = None,
        metric_creators: List["OIDIQMetricCreator"] | Dict[str, "OIDIQMetricCreator"] | None = None,
        verbose: bool = False,
    ):
        raw_imgs = []
        for img in imgs:
            if isinstance(img, str):
                img = np.array(Image.open(img))
                img = img[..., :3]  # Ensure RGB
            elif not isinstance(img, np.ndarray):
                raise ValueError(f"Invalid image input: {img}, type: {type(img)}")
            raw_imgs.append(img)
            
        super().__init__(raw_imgs, preprocessors, metric_creators, verbose)
        self._cache: Dict[str, Any] = {}
        self.batch_size = len(raw_imgs)
        self._requests: List[str] = []
        self._save_requests = False

    def _is_cached(self, name: str) -> bool:
        if self._is_save_requests():
            self._get_requests().append(name)
            return False
        return name in self._cache and all(v is not None for v in self._cache[name])

    def _get_cache(self, name: str) -> Any:
        return self._cache[name]

    def _set_cache(self, name: str, value: Any) -> None:
        self._cache[name] = value

    def _delete_cached(self, name: str) -> None:
        del self._cache[name]

    def reset_cache(self) -> None:
        self._cache = {}

    def _get_all_cached_keys(self) -> Iterable[str]:
        return self._cache.keys()
    
    def _is_save_requests(self) -> bool:
        return self._save_requests
    
    def _set_save_requests(self, value: bool) -> None:
        self._save_requests = value

    def _get_requests(self) -> List[str]:
        return self._requests

    @overload
    def __getitem__(self, idx: int) -> "OIDIQBaseSingleSession": ...
    @overload
    def __getitem__(self, idx: slice) -> "OIDIQBaseBatchSession": ...

    def __getitem__(self, idx: int | slice) -> OIDIQBaseBatchSession | OIDIQBaseSingleSession:
        if isinstance(idx, int):
            return OIDIQSingleBatchSessionWrapper(self, idx)
        elif isinstance(idx, slice):
            indices = list(range(*idx.indices(self.batch_size)))
            return OIDIQBatchSessionSlice(self, indices)
        else:
            raise TypeError("Index must be an integer or a slice.")

    def __len__(self) -> int:
        return self.batch_size


class OIDIQSingleBatchSessionWrapper(OIDIQBaseSingleSession):
    def __init__(
        self,
        wrapped_session: OIDIQBaseBatchSession,
        idx: int,
    ):
        self.wrapped_session = wrapped_session
        self.idx = idx
        super().__init__(
            wrapped_session.raw_img[idx],
            wrapped_session.preprocessors,
            wrapped_session.metric_creators,
            wrapped_session.verbose,
        )


    def _is_cached(self, name: str) -> bool:
        if self._is_save_requests():
            self._get_requests().append(name)
            return False
        return self.wrapped_session._is_cached(name) and self.wrapped_session._get_cache(name)[self.idx] is not None

    def _get_cache(self, name: str) -> Any:
        return self.wrapped_session._get_cache(name)[self.idx]

    def _set_cache(self, name: str, value: Any) -> None:
        if not self.wrapped_session._is_cached(name):
            self.wrapped_session._set_cache(name, [None] * len(self.wrapped_session))
        self.wrapped_session._get_cache(name)[self.idx] = value

    def _delete_cached(self, name: str) -> None:
        if self.wrapped_session._is_cached(name):
            self.wrapped_session._get_cache(name)[self.idx] = None

    def reset_cache(self) -> None:
        for k in self.wrapped_session._get_all_cached_keys():
            self.wrapped_session._get_cache(k)[self.idx] = None

    def _get_all_cached_keys(self) -> Iterable[str]:
        return self.wrapped_session._get_all_cached_keys()
    
    def _is_save_requests(self) -> bool:
        return self.wrapped_session._is_save_requests()
    
    def _set_save_requests(self, value: bool) -> None:
        self.wrapped_session._set_save_requests(value)
    
    def _get_requests(self) -> List[str]:
        return self.wrapped_session._get_requests()

    def to_batch(self) -> OIDIQBaseBatchSession:
        return self.wrapped_session[self.idx : self.idx + 1]


class OIDIQBatchSessionSlice(OIDIQBaseBatchSession):
    def __init__(
        self,
        wrapped_session: OIDIQBaseBatchSession,
        indices: List[int],
    ):
        self.wrapped_session = wrapped_session
        self.indices = indices
        super().__init__(
            [wrapped_session.raw_img[idx] for idx in indices],
            wrapped_session.preprocessors,
            wrapped_session.metric_creators,
            wrapped_session.verbose,
        )
        

    def _is_cached(self, name: str) -> bool:
        if self._is_save_requests():
            self._get_requests().append(name)
            return False
        return self.wrapped_session._is_cached(name) and all(
            self.wrapped_session._get_cache(name)[idx] is not None for idx in self.indices
        )

    def _get_cache(self, name: str) -> Any:
        cached = self.wrapped_session._get_cache(name)
        if all(isinstance(cached[idx], np.ndarray) for idx in self.indices):
            return np.array([cached[idx] for idx in self.indices])
        return [cached[idx] for idx in self.indices]

    def _set_cache(self, name: str, value: Any) -> None:
        if not self.wrapped_session._is_cached(name):
            self.wrapped_session._set_cache(name, [None] * len(self.wrapped_session))
        cached = self.wrapped_session._get_cache(name)
        for i, idx in enumerate(self.indices):
            cached[idx] = value[i]

    def _delete_cached(self, name: str) -> None:
        if self.wrapped_session._is_cached(name):
            cached = self.wrapped_session._get_cache(name)
            for idx in self.indices:
                cached[idx] = None

    def reset_cache(self) -> None:
        for k in self.wrapped_session._get_all_cached_keys():
            cached = self.wrapped_session._get_cache(k)
            for idx in self.indices:
                cached[idx] = None

    def _get_all_cached_keys(self) -> Iterable[str]:
        return self.wrapped_session._get_all_cached_keys()

    def _is_save_requests(self) -> bool:
        return self.wrapped_session._is_save_requests()
    
    def _set_save_requests(self, value: bool) -> None:
        self.wrapped_session._set_save_requests(value)

    def _get_requests(self) -> List[str]:
        return self.wrapped_session._get_requests()

    @overload
    def __getitem__(self, idx: int) -> "OIDIQBaseSingleSession": ...
    @overload
    def __getitem__(self, idx: slice) -> "OIDIQBaseBatchSession": ...

    def __getitem__(self, idx: int | slice) -> OIDIQBaseBatchSession | OIDIQBaseSingleSession:
        if isinstance(idx, int):
            return OIDIQSingleBatchSessionWrapper(self.wrapped_session, self.indices[idx])
        elif isinstance(idx, slice):
            new_indices = self.indices[idx]
            return OIDIQBatchSessionSlice(self.wrapped_session, new_indices)
        else:
            raise TypeError("Index must be an integer or a slice.")

    def __len__(self) -> int:
        return len(self.indices)


class OIDIQSession(OIDIQBaseSingleSession):
    def __init__(
        self,
        img: np.ndarray | str,
        preprocessors: List["OIDIQPreprocessor"] | Dict[str, "OIDIQPreprocessor"] | None = None,
        metric_creators: List["OIDIQMetricCreator"] | Dict[str, "OIDIQMetricCreator"] | None = None,
        verbose: bool = False,
    ):
        if isinstance(img, str):
            img = np.array(Image.open(img))
            img = img[..., :3]  # Ensure RGB
        super().__init__(img, preprocessors, metric_creators, verbose)
        self._cache: Dict[str, Any] = {}
        self._requests: List[str] = []
        self._save_requests = False

    def _is_cached(self, name: str) -> bool:
        if self._is_save_requests():
            self._get_requests().append(name)
            return False
        return name in self._cache

    def _get_cache(self, name: str) -> Any:
        return self._cache[name]

    def _set_cache(self, name: str, value: Any) -> None:
        self._cache[name] = value

    def _delete_cached(self, name: str) -> None:
        del self._cache[name]

    def reset_cache(self) -> None:
        self._cache = {}

    def _get_all_cached_keys(self) -> Iterable[str]:
        return self._cache.keys()
    
    def _is_save_requests(self) -> bool:
        return self._save_requests
    
    def _set_save_requests(self, value: bool) -> None:
        self._save_requests = value

    def _get_requests(self) -> List[str]:
        return self._requests


    def to_batch(self) -> OIDIQBaseBatchSession:
        return OIDIQSessionBatchWrapper(self)


class OIDIQSessionBatchWrapper(OIDIQBaseBatchSession):
    def __init__(
        self,
        wrapped_session: OIDIQBaseSingleSession,
    ):
        self.wrapped_session = wrapped_session
        raw_imgs = list([wrapped_session.raw_img])
        super().__init__(
            raw_imgs,
            wrapped_session.preprocessors,
            wrapped_session.metric_creators,
            wrapped_session.verbose,
        )

    def _is_cached(self, name: str) -> bool:
        if self._is_save_requests():
            self._get_requests().append(name)
            return False
        return self.wrapped_session._is_cached(name)

    def _get_cache(self, name: str) -> List[Any]:
        return [self.wrapped_session._get_cache(name)]

    def _set_cache(self, name: str, value: List[Any]) -> None:
        self.wrapped_session._set_cache(name, value[0])

    def _delete_cached(self, name: str) -> None:
        self.wrapped_session._delete_cached(name)

    def reset_cache(self) -> None:
        self.wrapped_session.reset_cache()

    def _get_all_cached_keys(self) -> Iterable[str]:
        return self.wrapped_session._get_all_cached_keys()
    
    def _is_save_requests(self) -> bool:
        return self.wrapped_session._is_save_requests()
    
    def _set_save_requests(self, value: bool) -> None:
        self.wrapped_session._set_save_requests(value)

    def _get_requests(self) -> List[str]:
        return self.wrapped_session._get_requests()

    @overload
    def __getitem__(self, idx: int) -> "OIDIQBaseSingleSession": ...
    @overload
    def __getitem__(self, idx: slice) -> "OIDIQBaseBatchSession": ...

    def __getitem__(self, idx: int | slice) -> OIDIQBaseBatchSession | OIDIQBaseSingleSession:
        if isinstance(idx, int):
            if idx == 0:
                return self.wrapped_session
            else:
                raise IndexError("Index out of range for single session batch wrapper.")
        elif isinstance(idx, slice):
            if idx.start in (None, 0) and idx.stop in (None, 1) and idx.step in (None, 1):
                return self
            else:
                raise IndexError("Index out of range for single session batch wrapper.")
        else:
            raise TypeError("Index must be an integer or a slice.")

    def __len__(self) -> int:
        return 1


R = TypeVar("R")
C = TypeVar("C", bound=OIDIQConfig)


class OIDIQBaseExecutor(Generic[R, C], ABC):
    def __init__(self, config: Optional[Dict] = None, overwrite_target: Optional[str] = None, **overwrite_config):
        self._create_functions: Dict[
            str,
            Callable[[OIDIQBaseSession], Any]
            | Callable[[OIDIQBaseSession, C], Any]
            | Callable[[OIDIQBaseSession, C, C], Any]
            | Callable[[OIDIQBaseSession, C, C, C], Any]
            | Callable[[OIDIQBaseSession, C, C, C, C], Any],
        ] = {}
        self._configs: Dict[str, C] = {}
        self._name_to_key: Dict[str, str] = {}
        super().__init__()
        for func in dir(self):
            func_obj = getattr(self, func)
            

            if hasattr(func_obj, "_creates") or hasattr(func_obj, "_creates_see_class"):
                for create in self._get_func_crates(func_obj) or []:
                    self._create_functions[create] = func_obj
                    self._configs[create] = self._create_config(_is_placeholder=True)
            config_keys = self._get_config_keys(func_obj)
            if config_keys:
                for key in config_keys:
                    self._configs[key] = self._create_config()

        if config is not None:
            for key in self._configs.keys():
                if key in config:
                    self._configs[key].update(deep_copy_dict(config[key]))

        if len(overwrite_config) > 0:
            if overwrite_target == "*":
                for key in self._configs.keys():
                    self._configs[key].update(deep_copy_dict(overwrite_config))
            else:
                key = self._get_config_key(overwrite_target)
                self._configs[key].update(deep_copy_dict(overwrite_config))

        self._call_init_config()

        for key, cfg in self._configs.items():
            if "name" in cfg:
                self._name_to_key[cfg["name"]] = key

    @abstractmethod
    def _create_config(self, cfg: Optional[Dict] = None, _is_placeholder: bool = False) -> C:
        pass

    def _get_func_crates(self, func: Callable) -> List[str] | None:
        if hasattr(func, "_creates"):
            return list(func._creates)
        elif hasattr(func, "_creates_see_class"):
            return list(self._creates) # type: ignore
        else:
            return None

    def _get_config_keys(
        self,
        func: (
            Callable[[OIDIQBaseSession], Any]
            | Callable[[OIDIQBaseSession, C], Any]
            | Callable[[OIDIQBaseSession, C, C], Any]
            | Callable[[OIDIQBaseSession, C, C, C], Any]
            | Callable[[OIDIQBaseSession, C, C, C, C], Any]
        ),
    ) -> List[str] | None:
        if hasattr(func, "_config_keys"):
            return list(func._config_keys)
        return self._get_func_crates(func)

    def _execute(
        self,
        session: OIDIQBaseSession,
        func: (
            Callable[[OIDIQBaseSession], Any]
            | Callable[[OIDIQBaseSession, C], Any]
            | Callable[[OIDIQBaseSession, C, C], Any]
            | Callable[[OIDIQBaseSession, C, C, C], Any]
            | Callable[[OIDIQBaseSession, C, C, C, C], Any]
        ),
    ) -> Dict[str, List[R] | R]:
        config_keys = self._get_config_keys(func)
        if config_keys is None:
            raise ValueError(f"Failed to get config keys for function {func.__name__} in {self.__class__.__name__}.")
        configs = [self._configs[key] for key in config_keys]

        batch_size = self._get_batch_size(func, configs)

        if batch_size is not None:
            if isinstance(session, OIDIQBaseSingleSession):
                r = self._execute_batch(session.to_batch(), func, configs)
                return {k: v[0] for k, v in r.items()}
            elif not isinstance(session, OIDIQBaseBatchSession):
                return self._execute_batch(session, func, configs)
            r_list = []

            for start_idx in range(0, len(session), batch_size):
                end_idx = start_idx + batch_size
                end_idx = min(end_idx, len(session))
                s = session[start_idx:end_idx]
                r_list.append(self._execute_batch(s, func, configs))
            r = {}
            for k in r_list[0].keys():
                r[k] = []
                for r_item in r_list:
                    r[k].extend(r_item[k])
            return r
        else:
            if isinstance(session, OIDIQBaseSingleSession):
                return self._execute_batch(session, func, configs)
            elif isinstance(session, OIDIQBaseBatchSession):
                r_list = []
                for i in range(len(session)):
                    r_list.append(self._execute_batch(session[i], func, configs))
                r = {}
                for k in r_list[0].keys():
                    r[k] = []
                    for r_item in r_list:
                        r[k].append(r_item[k])
                return r
            else:
                return self._execute_batch(session, func, configs)

    def _execute_batch(
        self,
        session: OIDIQBaseSession,
        func: Callable,
        configs: List[C],
    ):
        r = func(session, *configs)
        creates = self._get_func_crates(func)
        if creates is None:
            raise ValueError(f"Function {func.__name__} in {self.__class__.__name__} does not have '_creates' or '_creates_see_class' attribute.")
        if len(creates) == 1:
            key = self._get_create_names(func, configs)[0]
            return {key: r}
        if not isinstance(r, tuple) or len(r) != len(creates):
            raise ValueError(f"Expected {len(creates)} results from {func.__name__} in {self.__class__.__name__}")
        keys = self._get_create_names(func, configs)
        return {k: v for k, v in zip(keys, r)}

    def _get_create_names(self, func: Callable, configs) -> List[str]:
        creates = self._get_func_crates(func)
        if creates is None:
            raise ValueError(f"Function {func.__name__} in {self.__class__.__name__} does not have '_creates' or '_creates_see_class' attribute.")
        create_configs = [self._configs[k] for k in creates]
        names = []
        if len(create_configs) == len(configs):
            for k, create_config, config in zip(creates, create_configs, configs):
                if "name" in create_config:
                    names.append(create_config["name"])
                elif "name" in config:
                    names.append(config["name"])
                else:
                    names.append(k)
            return names
        else:
            for k, create_config in zip(creates, create_configs):
                if "name" in create_config:
                    names.append(create_config["name"])
                else:
                    names.append(k)
            return names

    def copy(
        self, config: Optional[Dict[str, Dict]] = None, overwrite_target: Optional[str] = None, **overwrite_config
    ) -> "OIDIQBaseExecutor[R, C]":
        return self.__class__(config, overwrite_target, **overwrite_config)

    def creates(self) -> Set[str]:
        return (set(self._create_functions.keys()) - set(self._name_to_key.values())) | set(self._name_to_key.keys())

    def get(self, session: OIDIQBaseSession | Any, *names: str) -> Dict[str, R] | Dict[str, List[R]]:
        if not names:
            r = {}
            for func in set(self._create_functions.values()):
                r.update(self._execute(session, func))
            return r

        mapped_names = []
        for name in names:
            mapped_name = self._name_to_key.get(name, name)
            if mapped_name not in self._create_functions:
                raise KeyError(f"Name '{name}' not found in executor {self.__class__.__name__}.")
            mapped_names.append(mapped_name)
        r = {}
        queue = mapped_names.copy()
        called_functions: Set[Callable] = set()
        while queue:
            name = queue.pop(0)
            func = self._create_functions[name]
            if func in called_functions:
                continue
            result = self._execute(session, func)
            r.update(result)
            called_functions.add(func)
        return r

    def init_config(self, *configs: C):
        pass

    def _get_config_key(self, config: str | None = None) -> str:
        if config is not None:
            if config not in self._configs:
                raise KeyError(f"Config '{config}' not found in executor {self.__class__.__name__}.")
            return config
        elif len(self._configs) != 1:
            filtered_keys = [k for k, v in self._configs.items() if not v._is_placeholder]
            if len(filtered_keys) == 1:
                return filtered_keys[0]
            raise ValueError("When getting config you must specify the config name if there are multiple configs.")
        else:
            return list(self._configs.keys())[0]

    def _get_batch_size(self, func, configs: List[C]) -> int | None:
        if not hasattr(func, "_batch_size") or func._batch_size is None:
            return None
        min_batch_size = None
        for cfg in configs:
            if "batch_size" in cfg:
                if min_batch_size is None:
                    min_batch_size = cfg["batch_size"]
                else:
                    min_batch_size = min(min_batch_size, cfg["batch_size"])
        create_configs = [self._configs[k] for k in func._creates]
        for cfg in create_configs:
            if "batch_size" in cfg:
                if min_batch_size is None:
                    min_batch_size = cfg["batch_size"]
                else:
                    min_batch_size = min(min_batch_size, cfg["batch_size"])
        return min_batch_size if min_batch_size is not None else func._batch_size

    def _call_init_config(self):
        if hasattr(self.init_config, "_config_keys"):
            config_keys = self.init_config._config_keys  # type: ignore
            configs = [self._configs[key] for key in config_keys]
            self.init_config(*configs)

    def update_config(self, config: str | None = None, call_init_config: bool = True, **overwrite_config):
        if len(overwrite_config) == 0:
            return
        if config == "*":
            for key in self._configs.keys():
                self.update_config(key, call_init_config=False, **overwrite_config)
            return
        if "name" in overwrite_config:
            key = self._get_config_key(config)
            if "name" in self._configs[key]:
                old_name = self._configs[key]["name"]
                del self._name_to_key[old_name]
        key = self._get_config_key(config)
        self._configs[key].update(deep_copy_dict(overwrite_config))
        if "name" in overwrite_config:
            self._name_to_key[overwrite_config["name"]] = key

        if call_init_config:
            self._call_init_config()

    def add_name_postfix(self, postfix: str):
        for func in set(self._create_functions.values()):
            config_keys = self._get_config_keys(func)
            if config_keys is None:
                continue
            configs = [self._configs[key] for key in config_keys]
            cur_names = self._get_create_names(func, configs)
            for creates, cur_name in zip(func._creates, cur_names):
                self._configs[creates]["name"] = cur_name + postfix
                self._name_to_key[cur_name + postfix] = creates
                if cur_name in self._name_to_key:
                    del self._name_to_key[cur_name]

    def get_config(self, config: str | None = None) -> C:
        key = self._get_config_key(config)
        return self._configs[key]


class OIDIQPreprocessor(OIDIQBaseExecutor[Any, OIDIQConfig]):

    def _create_config(self, cfg=None, _is_placeholder: bool = False) -> OIDIQConfig:
        return OIDIQConfig(cfg, _is_placeholder=_is_placeholder)

    @staticmethod
    def from_function(
        func: Callable,
        creates: Tuple[str, ...] | str,
        batch_size: Optional[int] = None,
        overwrite_target: Optional[str] = None,
        **overwrite_config,
    ) -> "OIDIQPreprocessor":
        if isinstance(creates, str):
            creates = (creates,)
        if len(inspect.signature(func).parameters) == 1:

            def wrapper(session, *configs):
                return func(session)  # type: ignore

            _func = wrapper
        else:
            _func = func

        class FunctionPreprocessor(OIDIQPreprocessor):
            @creates_alias(*creates)
            @batching(batch_size)
            def execute(self, session: OIDIQSession, *configs: OIDIQConfig) -> Any:
                return _func(session, *configs)

        return FunctionPreprocessor(overwrite_target=overwrite_target, **overwrite_config)


class OIDIQMetricCreator(OIDIQBaseExecutor[QualityMetric, QualityMetricConfig]):

    def _create_config(self, cfg=None, _is_placeholder: bool = False) -> QualityMetricConfig:
        return QualityMetricConfig(cfg, _is_placeholder=_is_placeholder)

    def get(
        self, session: OIDIQBaseSession | Any, *names: str
    ) -> Dict[str, QualityMetric] | Dict[str, List[QualityMetric]]:
        r = super().get(session, *names)
        for k, v in r.items():
            if isinstance(v, list):
                if v[0].name == QualityMetricConfig.DEFAULT_NAME:
                    for i in range(len(v)):
                        v[i].name = k
            else:
                if v.name == QualityMetricConfig.DEFAULT_NAME:
                    v.name = k
        return r

    @staticmethod
    def from_function(
        func: Callable,
        creates: str | Tuple[str, ...],
        description: str | Tuple[str, ...] | None = None,
        batch_size: Optional[int] = None,
        overwrite_target: Optional[str] = None,
        **overwrite_config,
    ) -> "OIDIQMetricCreator":
        if isinstance(creates, str):
            creates = (creates,)
        if description is not None:
            if isinstance(description, str):
                description = (description,)
            if len(description) != len(creates):
                raise ValueError("Length of description must match length of creates.")

        if len(inspect.signature(func).parameters) == 1:
            raise ValueError("Metric function must accept at least one config parameter to create QualityMetric.")
        else:
            _func = func

        class FunctionMetricCreator(OIDIQMetricCreator):
            @creates_alias(*creates)
            @batching(batch_size)
            def execute(self, session: OIDIQBaseSession, *configs: QualityMetricConfig) -> Any:
                return _func(session, *configs)

        config = {}
        for name, desc in zip(creates, description or ["" for _ in creates]):
            config[name] = {"name": name, "description": desc}

        return FunctionMetricCreator(config, overwrite_target=overwrite_target, **overwrite_config)
