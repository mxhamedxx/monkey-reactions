import time

import cv2

from detector import ExpressionDetector
from reactions import ReactionImages
from stats import ReactionStats

from ui import (
    draw_controls,
    draw_debug_info,
    draw_main_info,
    draw_mouth_points,
    draw_stats
)


def main():
    camera = cv2.VideoCapture(0)

    window_name = "Monkey Reaction"
    fullscreen = False

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    if not camera.isOpened():
        print("Could not open webcam.")
        return

    detector = ExpressionDetector()
    reaction_images = ReactionImages()
    stats = ReactionStats()

    debug_mode = False

    previous_time = time.time()

    fps = 0

    screenshot_count = 1

    print("Monkey Reaction started!")
    print("Q = Quit")
    print("D = Toggle debug mode")
    print("S = Screenshot")
    print("R = Reset statistics")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        frame = cv2.flip(
            frame,
            1
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        data = detector.process(
            rgb_frame
        )

        # Update reaction statistics
        stats.update(
            data["expression"]
        )

        # FPS
        current_time = time.time()

        delta_time = (
            current_time - previous_time
        )

        if delta_time > 0:
            current_fps = 1 / delta_time

            fps = (
                0.9 * fps
                +
                0.1 * current_fps
            )

        previous_time = current_time

        draw_main_info(
            frame,
            data["expression"],
            fps
        )

        if debug_mode:
            draw_mouth_points(
                frame,
                data["mouth_points"]
            )

            draw_debug_info(
                frame,
                data
            )

        frame_height, frame_width, _ = (
            frame.shape
        )

        monkey = reaction_images.get(
            data["expression"],
            frame_width,
            frame_height
        )

        # Draw statistics on monkey side
        draw_stats(
            monkey,
            stats
        )

        combined = cv2.hconcat([
            frame,
            monkey
        ])

        draw_controls(
            combined
        )

        cv2.imshow(
            window_name,
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

        elif key == ord("r"):
            stats.reset()

            print(
                "Reaction statistics reset."
            )

        elif key == ord("f"):
            fullscreen = not fullscreen

            if fullscreen:
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN
                )

                print("Fullscreen mode enabled.")

            else:
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL
                )

                print("Fullscreen mode disabled.")

    detector.close()

    camera.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()