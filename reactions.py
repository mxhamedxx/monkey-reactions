from pathlib import Path

import cv2


class ReactionImages:
    def __init__(self):
        base_path = Path(__file__).resolve().parent

        assets_path = base_path / "assets"

        self.images = {
            "THINKING": self.load_image(
                assets_path / "thinking.png"
            ),
            "SMILING": self.load_image(
                assets_path / "smiling.png"
            ),
            "SHOCKED": self.load_image(
                assets_path / "shocked.png"
            )
        }

    def load_image(self, path):
        image = cv2.imread(str(path))

        if image is None:
            raise FileNotFoundError(
                f"Could not load {path}"
            )

        return image

    def get(self, expression, width, height):
        image = self.images[expression]

        return cv2.resize(
            image,
            (width, height)
        )