import cv2


class ReactionImages:
    def __init__(self):
        self.images = {
            "THINKING": self.load_image(
                "assets/thinking.png"
            ),
            "SMILING": self.load_image(
                "assets/smiling.png"
            ),
            "SHOCKED": self.load_image(
                "assets/shocked.png"
            )
        }

    def load_image(self, path):
        image = cv2.imread(path)

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