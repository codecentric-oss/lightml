from dataclasses import dataclass

from canaryml.datadescriptions import DataDescription


@dataclass
class CategoryDataDescription(DataDescription):
    names: list[str]

    def get_category_index(self, name: str) -> int:
        return self.names.index(name)

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self):
        return f"CategoryDataDescription(names={self.names})"

    def __iter__(self):
        return iter(self.names)

    def __str__(self):
        return f"Names: {self.names}"

    def __eq__(self, other: "DataDescription") -> bool:
        if isinstance(other, CategoryDataDescription):
            return self.names == other.names
        return False

    def __getitem__(self, index: int) -> str:
        return self.names.[index]
