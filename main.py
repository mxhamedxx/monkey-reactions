import cv2
import mediapipe as mp

def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open webcam.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face landmarks
        results = face_mesh.process(rgb_frame)

        # If a face was detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS
                )

        cv2.imshow("Monkey Reaction", frame)

        # Press Q to close
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    face_mesh.close()
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
