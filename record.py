from dataclasses import dataclass
from typing import Generic, Iterable, Iterator, TypeVar


T = TypeVar('T')
@dataclass
class Record(Generic[T]):
    record: list[T]

    def __init__(self, iterable: Iterable[T] | Iterator[T]):
        self.iter = iter(iterable)
        self.record = []

    def __iter__(self):
        yield from self.record
        for item in self.iter:
            self.record.append(item)
            yield item