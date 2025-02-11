from dataclasses import dataclass

from canaryml.datadescriptions.data_description import DataDescription
from canaryml.utils.image import ImageSize


@dataclass
class ImageDataDescription(DataDescription):
    image_size: ImageSize
    channel_names: list[str]

    def get_tensor_shape(
        self, channel_count_first: bool = False
    ) -> tuple[int, int, int]:
        """Returns the 3-dim shape of the input tensor [height, width, channel_count]"""
        if channel_count_first:
            return (self.get_channel_count(),) + self.image_size.to_numpy_shape()
        return self.image_size.to_numpy_shape() + (self.get_channel_count(),)

    def get_channel_count(self) -> int:
        """Returns the number of channels"""
        return len(self.channel_names)

    def __repr__(self):
        return f"ImageDataDescription(image_size={self.image_size}, channel_names={self.channel_names})"

    def __len__(self):
        return self.get_channel_count()

    def __iter__(self):
        return iter(self.channel_names)

    def __str__(self):
        return f"Image Size: {self.image_size}, Channel Names: {self.channel_names}"

    def __eq__(self, other: "DataDescription") -> bool:
        if isinstance(other, ImageDataDescription):
            return (
                self.image_size == other.image_size
                and self.channel_names == other.channel_names
            )
        return False
