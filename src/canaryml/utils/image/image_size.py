# Copyright (c) codecentric AG 2025. All Rights Reserved.
#
# Licensed under the MIT License. See the LICENSE file in the project root for more information.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# 1. The above copyright notice and this permission notice shall be included in
#    all copies or substantial portions of the Software.
#
# 2. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

from dataclasses import dataclass
from math import isclose

import numpy as np
from PIL import Image

IDENTITY_SCALE = 1.0


class ImageSizeDivisionError(Exception):
    """Error for if two ImageSizes have different aspect ratios"""


@dataclass
class ImageSize:
    """Class to represent the size of image"""

    width: int
    height: int

    def to_numpy_shape(self) -> tuple[int, int]:
        """tuple with height, width"""
        return self.height, self.width

    def to_pil_size(self) -> tuple[int, int]:
        """tuple with width, height"""
        return self.width, self.height

    def __str__(self) -> str:
        """string with width, height"""
        return f"[{self.width},{self.height}]"

    def __truediv__(self, divider: float) -> "ImageSize":
        """Divides the ImageSize by a given division factor"""
        return ImageSize(round(self.width / divider), round(self.height / divider))

    def __mul__(self, scale: float) -> "ImageSize":
        """Multiplies the ImageSize by a given scale factor"""
        return ImageSize(round(self.width * scale), round(self.height * scale))

    def __eq__(self, other: "ImageSize"):
        """Checks if two ImageSizes are equal"""
        return (
            other is not None
            and isinstance(other, ImageSize)
            and self.width == other.width
            and self.height == other.height
        )

    def get_division_factor(self, other: "ImageSize") -> float:
        """Calculates the scale factor of two ImageSizes and returns it"""
        scale = self.width / other.width
        if not isclose(scale, self.height / other.height):
            raise ImageSizeDivisionError(
                f"The ImageSizes have different aspect ratios"
                f"width_ratio: {scale} height_ratio: "
                f"{self.height / other.height}"
            )
        return scale

    def np_array_has_same_size(self, np_array: np.ndarray) -> bool:
        """Checks if the np_array has the same size as the ImageSize"""
        return self.width == np_array.shape[1] and self.height == np_array.shape[0]

    def __le__(self, other: "ImageSize") -> bool:
        """Compares ImageSizes with the same aspect ratio"""
        division_factor = self.get_division_factor(other)
        return division_factor <= IDENTITY_SCALE

    def __lt__(self, other: "ImageSize") -> bool:
        """Compares ImageSizes with the same aspect ratio"""
        division_factor = self.get_division_factor(other)
        return division_factor < IDENTITY_SCALE

    def create_with_same_aspect_ratio(  # QUEST: what should be done with both given?
        self, *, width: int | None = None, height: int | None = None
    ) -> "ImageSize":
        """
        Creates an ImageSize with the same aspect ratio given either width or height
        of the target ImageSize

        Args:
            width: width of the target ImageSize
            height: height of the target ImageSize

        Returns:
            new ImageSize, based on height or width, with the same aspect ratio as the
            original Image Size
        """
        if width is None and height is None:
            raise ValueError("Either width or height must be set")
        if width is not None and height is not None:
            cur_image_size = ImageSize(width, height)
            self.get_division_factor(cur_image_size)
        if width is not None:
            cur_image_size = ImageSize(width, round(width * self.height / self.width))
        else:
            cur_image_size = ImageSize(round(height * self.width / self.height), height)

        return cur_image_size

    @classmethod
    def from_numpy_shape(cls, shape: tuple[int, int]) -> "ImageSize":
        """Creates an ImageSize from a numpy shape"""
        return cls(shape[1], shape[0])

    @classmethod
    def from_pil_size(cls, size: tuple[int, int]) -> "ImageSize":
        """Creates an ImageSize from a PIL size"""
        return cls(size[0], size[1])

    @classmethod
    def from_pil_image(cls, image: Image.Image) -> "ImageSize":
        """Creates an ImageSize from a PIL Image"""
        return cls(image.width, image.height)


def create_image_size(
    argument: str | list[int] | tuple[int, int] | dict,
) -> ImageSize:
    """
    Creates an ImageSize with:
    - a list of two ints [width, height]
    - a Tuple of two ints (width, height)
    - a str of form width,height, which will be parsed

    Args:
        argument: List, Tuple or String containing width and height information

    Returns:
        ImageSize with given width and height
    """
    if isinstance(argument, str):
        try:
            width, height = argument.split(",")
        except ValueError as error:
            raise Exception(f"ImageSize couldn't be parsed: {argument}") from error
    elif isinstance(argument, dict):
        width = argument["width"]
        height = argument["height"]
    else:
        width, height = argument
    return ImageSize(int(width), int(height))
