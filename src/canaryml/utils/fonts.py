"""module to load fonts from assets"""
from pathlib import Path

from PIL import ImageFont


def get_font(font_name: str, font_size: int = 50) -> ImageFont.FreeTypeFont:
    """Returns a randomly selected `FreeTypeFont` from ten predefined font names"""
    font_path = f"{Path(__file__).parent.resolve()}/assets/fonts/{font_name}"
    return ImageFont.truetype(font=font_path, size=font_size)
