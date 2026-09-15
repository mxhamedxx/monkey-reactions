import math

from collections import deque, Counter

import mediapipe as mp


SMILE_WIDTH_THRESHOLD = 0.39
SHOCKED_OPEN_THRESHOLD = 0.075
SHOCKED_SHAPE_THRESHOLD = 0.30

HISTORY_SIZE = 7


class ExpressionDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.expression_history = deque(
            maxlen=HISTORY_SIZE
        )

    def distance(self, point1, point2):
        return math.sqrt(
            (point1.x - point2.x) ** 2
            +
            (point1.y - point2.y) ** 2
        )

    def classify_expression(
        self,
        mouth_width_ratio,
        mouth_open_ratio,
        mouth_shape_ratio
    ):
        if (
            mouth_width_ratio > SMILE_WIDTH_THRESHOLD
            and mouth_shape_ratio < SHOCKED_SHAPE_THRESHOLD
        ):
            return "SMILING"

        if (
            mouth_open_ratio > SHOCKED_OPEN_THRESHOLD
            and mouth_shape_ratio >= SHOCKED_SHAPE_THRESHOLD
        ):
            return "SHOCKED"

        return "THINKING"

    def get_smoothed_expression(self):
        counts = Counter(
            self.expression_history
        )

        return counts.most_common(1)[0][0]

    def process(self, rgb_frame):
        results = self.face_mesh.process(
            rgb_frame
        )

        data = {
            "expression": "THINKING",
            "raw_expression": "THINKING",
            "mouth_width_ratio": 0,
            "mouth_open_ratio": 0,
            "mouth_shape_ratio": 0,
            "mouth_points": [],
            "face_detected": False
        }

        if results.multi_face_landmarks:
            data["face_detected"] = True

            face_landmarks = (
                results.multi_face_landmarks[0]
            )

            landmarks = face_landmarks.landmark

            left_mouth = landmarks[61]
            right_mouth = landmarks[291]

            upper_lip = landmarks[13]
            lower_lip = landmarks[14]

            left_face = landmarks[234]
            right_face = landmarks[454]

            mouth_width = self.distance(
                left_mouth,
                right_mouth
            )

            mouth_opening = self.distance(
                upper_lip,
                lower_lip
            )

            face_width = self.distance(
                left_face,
                right_face
            )

            mouth_width_ratio = (
                mouth_width / face_width
            )

            mouth_open_ratio = (
                mouth_opening / face_width
            )

            mouth_shape_ratio = (
                mouth_opening / mouth_width
            )

            raw_expression = (
                self.classify_expression(
                    mouth_width_ratio,
                    mouth_open_ratio,
                    mouth_shape_ratio
                )
            )

            data.update({
                "raw_expression": raw_expression,
                "mouth_width_ratio": mouth_width_ratio,
                "mouth_open_ratio": mouth_open_ratio,
                "mouth_shape_ratio": mouth_shape_ratio,
                "mouth_points": [
                    left_mouth,
                    right_mouth,
                    upper_lip,
                    lower_lip
                ]
            })

        self.expression_history.append(
            data["raw_expression"]
        )

        data["expression"] = (
            self.get_smoothed_expression()
        )

        return data

    def close(self):
        self.face_mesh.close()