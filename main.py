import cv2
import mediapipe as mp
import math
import time

from collections import deque, Counter


# Starting thresholds.
SMILE_WIDTH_THRESHOLD = 0.39
SHOCKED_SHAPE_THRESHOLD = 0.30
SHOCKED_OPEN_THRESHOLD = 0.075

HISTORY_SIZE = 7

def distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )

def classify_expression(mouth_width_ratio, mouth_open_ratio, mouth_shape_ratio):
    # wide mouth usually means smiling,
    # even if teeth make the mouth somewhat open
    if (
        mouth_width_ratio > SMILE_WIDTH_THRESHOLD
        and mouth_shape_ratio < SHOCKED_SHAPE_THRESHOLD
    ):
        return "SMILING"

    # Shocked mouth is both open and round/tall.
    if (
        mouth_open_ratio > SHOCKED_OPEN_THRESHOLD
        and mouth_open_ratio >= SHOCKED_SHAPE_THRESHOLD
    ):
        return "SHOCKED"

    return "THINKING"

def get_smooth_expression(history):
    counts = Counter(history)

    return counts.most_common(1)[0][0]

def load_reaction_images():
    thinking = cv2.imread("assets/thinking.png")
    smiling = cv2.imread("assets/smiling.png")
    shocked = cv2.imread("assets/shocked.png")

    if thinking is None:
        raise FileNotFoundError("Could not load assets/thinking.png")

    if smiling is None:
        raise FileNotFoundError("Could not load assets/smiling.png")

    if shocked is None:
        raise FileNotFoundError("Could not load assets/shocked.png")

    return {
        "THINKING": thinking,
        "SMILING": smiling,
        "SHOCKED": shocked
    }

def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

def draw_debug_points(frame, points):
    height, width, _ = frame.shape

    for point in points:
        x = int(point.x * width)
        y = int(point.y * height)

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open webcam.")
        return

    reaction_images = load_reaction_images()

    # Stores recent expression predictions
    expression_history = deque(
        maxlen=HISTORY_SIZE
    )

    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    debug_mode = False

    previous_time = time.time()

    fps = 0

    screnshot_count = 1

    print("Monkey Reaction started!")
    print("Q = Quit")
    print("D = Toggle debug mode")
    print("S = Screenshot")

    print("Camera started! Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # MediaPipe expects RGB instead of OpenCV's BGR
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Detect face landmarks
        results = face_mesh.process(rgb_frame)

        raw_expression = "THINKING"

        mouth_width_ratio = 0
        mouth_open_ratio = 0
        mouth_shape_ratio = 0

        # If a face was detected
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            landmarks = face_landmarks.landmark

            # Important mouth landmarks
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]

            upper_lip = landmarks[13]
            lower_lip = landmarks[14]

            # Face sides used for normalization
            left_face = landmarks[234]
            right_face = landmarks[454]

            # distances
            mouth_width = distance(
                left_mouth,
                right_mouth
            )

            mouth_opening = distance(
                upper_lip,
                lower_lip
            )

            face_width = distance(
                left_face,
                right_face
            )

            # Normalize measurements
            mouth_width_ratio = mouth_width / face_width
            mouth_open_ratio = mouth_opening / face_width

            # Measures whether the mouth is wide like a small or tall like a shock
            mouth_shape_ratio = mouth_opening / mouth_width

            # Classify expression
            raw_expression = classify_expression(
                mouth_width_ratio,
                mouth_open_ratio,
                mouth_shape_ratio
            )

            if debug_mode:
                draw_debug_points(
                    frame,
                    [
                        left_mouth,
                        right_mouth,
                        upper_lip,
                        lower_lip
                    ]
                )

        expression_history.append(
            raw_expression
        )

        expression = get_smooth_expression(
            expression_history
        )

        # FPS calculation
        current_time = time.time()

        delta_time = (
            current_time - previous_time
        )

        if delta_time > 0:
            current_fps = 1 / delta_time

            # Smooth FPS slightly
            fps = (
                0.9 * fps +
                0.1 * current_fps
            )

        previous_time = current_time

        # Main reaction text
        cv2.putText(
            frame,
            f"Reaction: {expression}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )

        # FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Debug values
        if debug_mode:
            cv2.putText(
                frame,
                f"Width: {mouth_width_ratio:.3f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Open: {mouth_open_ratio:.3f}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Shape: {mouth_shape_ratio:.3f}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Raw: {raw_expression}",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        monkey = reaction_images[
            expression
        ]

        frame_height, frame_width, _ = (
            frame.shape
        )

        monkey = resize_image(
            monkey,
            frame_width,
            frame_height
        )

        combined = cv2.hconcat([
            frame,
            monkey
        ])

        # Controls
        cv2.putText(
            combined,
            "Q: Quit   D: Debug   S: Screenshot",
            (20, combined.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Monkey Reaction",
            combined
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("d"):
            debug_mode = not debug_mode

            print(
                f"Debug mode: {debug_mode}"
            )

        elif key == ord("s"):
            filename = (
                f"monkey_reaction_"
                f"{screenshot_count}.png"
            )

            cv2.imwrite(
                filename,
                combined
            )

            print(
                f"Screenshot saved: {filename}"
            )

            screenshot_count += 1

    face_mesh.close()
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
