import cv2


def draw_mouth_points(frame, points):
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


def draw_main_info(
    frame,
    expression,
    fps
):
    cv2.putText(
        frame,
        f"Reaction: {expression}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        3
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


def draw_debug_info(frame, data):
    cv2.putText(
        frame,
        f"Width: {data['mouth_width_ratio']:.3f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Open: {data['mouth_open_ratio']:.3f}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Shape: {data['mouth_shape_ratio']:.3f}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Raw: {data['raw_expression']}",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


def draw_controls(frame):
    cv2.putText(
        frame,
        "Q: Quit   D: Debug   S: Screenshot",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )