import cv2
import mediapipe as mp
import math

from collections import deque, Counter


# Starting thresholds.
# We can can tune these based on our webcam/face later
SHOCKED_THRESHOLD = 0.075
SMILE_WIDTH_THRESHOLD = 0.39
SMILE_OPEN_LIMIT = 0.065

HISTORY_SIZE = 7

def distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )

def classify_expression(mouth_width_ratio, mouth_open_ratio):
    if mouth_open_ratio > SHOCKED_THRESHOLD:
        return "SHOCKED"

    if (
        mouth_width_ratio > SMILE_WIDTH_THRESHOLD
        and mouth_open_ratio < SMILE_OPEN_LIMIT
    ):
        return "SMILING"

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

            # Classify expression
            raw_expression = classify_expression(
                mouth_width_ratio,
                mouth_open_ratio
            )

            # Display measurements
            cv2.putText(
                frame,
                f"Mouth Width: {mouth_width_ratio:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Mouth Open: {mouth_open_ratio:.3f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # Add current prediction to history
        expression_history.append(
            raw_expression
        )

        # Pick the most common recent reaction
        expression = get_smooth_expression(
            expression_history
        )

        cv2.putText(
            frame,
            f"Reaction: {expression}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )

        # Get the correct monkey
        monkey = reaction_images[expression]

        # Make monkey image same size as webcam frame
        frame_height, frame_width, _ = frame.shape

        monkey = resize_image(
            monkey,
            frame_width,
            frame_height
        )

        # Put webcam and monkey side-by-side
        combined = cv2.hconcat([
            frame,
            monkey,
        ])

        cv2.imshow(
            "Monkey Reaction",
            combined
        )
  
        # Press Q to close
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    face_mesh.close()
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
