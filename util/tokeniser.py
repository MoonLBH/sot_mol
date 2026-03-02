from __future__ import annotations

import pickle
import threading
from abc import ABC, abstractmethod
from typing import Optional, Union

from .functional import one_hot_encode

indicesT = Union[list[int], list[list[int]]]


PICKLE_PROTOCOL = 4


# *** Util functions ***


def _check_unique(obj_list, name="objects"):
    if len(obj_list) != len(set(obj_list)):
        raise RuntimeError(f"{name} cannot contain duplicates")


def _check_type_all(obj_list, exp_type, name="list"):
    for obj in obj_list:
        if not isinstance(obj, exp_type):
            raise TypeError(f"all objects in {name} must be instances of {exp_type}")


# *** Tokeniser Interface ***


class Tokeniser(ABC):
    """Interface for tokeniser classes"""

    @abstractmethod
    def tokenise(self, sentences: list[str]) -> Union[list[str], list[int]]:
        pass

    @classmethod
    @abstractmethod
    def from_vocabulary(cls, vocab: Vocabulary) -> Tokeniser:
        pass


# *** Tokeniser Implementations ***

# TODO


# *** Vocabulary Implementations ***


class Vocabulary:
    """Vocabulary class which maps tokens <--> indices"""

    def __init__(self, tokens: list[str]):
        _check_unique(tokens, "tokens list")

        token_idx_map = {token: idx for idx, token in enumerate(tokens)}
        idx_token_map = {idx: token for idx, token in enumerate(tokens)}

        self.token_idx_map = token_idx_map
        self.idx_token_map = idx_token_map

        # Just to be certain that vocab objects are thread safe
        self._vocab_lock = threading.Lock()

        # So that we can save this object without assuming the above dictionaries are ordered
        self._tokens = tokens

    @property
    def size(self) -> int:
        return len(self)

    def __len__(self) -> int:
        with self._vocab_lock:
            length = len(self.token_idx_map)

        return length

    def contains(self, token: str) -> bool:
        with self._vocab_lock:
            contains = token in self.token_idx_map

        return contains

    def tokens_from_indices(self, indices: list[int]) -> list[str]:
        _check_type_all(indices, int, "indices list")
        with self._vocab_lock:
            tokens = [self.idx_token_map[idx] for idx in indices]

        return tokens

    def indices_from_tokens(self, tokens: list[str], one_hot: Optional[bool] = False) -> indicesT:
        _check_type_all(tokens, str, "tokens list")

        with self._vocab_lock:
            indices = [self.token_idx_map[token] for token in tokens]

        if not one_hot:
            return indices

        one_hots = one_hot_encode(indices, len(self)).tolist()
        return one_hots

    def to_bytes(self) -> bytes:
        with self._vocab_lock:
            obj_bytes = pickle.dumps(self._tokens, protocol=PICKLE_PROTOCOL)

        return obj_bytes

    @staticmethod
    def from_bytes(data: bytes) -> Vocabulary:
        tokens = pickle.loads(data)
        return Vocabulary(tokens)