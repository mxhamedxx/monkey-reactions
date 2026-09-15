import cv2


def draw_panel(
    frame,
    top_left,
    bottom_right,
    alpha=0.55
):
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        top_left,
        bottom_right,
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )


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
    # Top header panel
    draw_panel(
        frame,
        (0, 0),
        (frame.shape[1], 95)
    )

    cv2.putText(
        frame,
        "MONKEY REACTION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Reaction: {expression}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"{fps:.1f} FPS",
        (frame.shape[1] - 115, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


def draw_debug_info(frame, data):
    draw_panel(
        frame,
        (15, 110),
        (310, 245)
    )

    cv2.putText(
        frame,
        "DEBUG",
        (30, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Width: {data['mouth_width_ratio']:.3f}",
        (30, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Open: {data['mouth_open_ratio']:.3f}",
        (30, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Shape: {data['mouth_shape_ratio']:.3f}",
        (30, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Raw: {data['raw_expression']}",
        (30, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


def draw_stats(frame, stats):
    height, width, _ = frame.shape

    panel_width = min(420, width - 40)

    draw_panel(
        frame,
        (20, height - 225),
        (20 + panel_width, height - 40),
        0.65
    )

    cv2.putText(
        frame,
        "SESSION STATS",
        (40, height - 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Thinking  {stats.durations['THINKING']:.1f}s",
        (40, height - 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        (
            f"Smiling   "
            f"{stats.durations['SMILING']:.1f}s"
            f"   x{stats.counts['SMILING']}"
        ),
        (40, height - 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        (
            f"Shocked   "
            f"{stats.durations['SHOCKED']:.1f}s"
            f"   x{stats.counts['SHOCKED']}"
        ),
        (40, height - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Session: {stats.get_session_time():.1f}s",
        (width - 180, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


def draw_controls(frame):
    height, width, _ = frame.shape

    draw_panel(
        frame,
        (0, height - 35),
        (width, height),
        0.65
    )

    cv2.putText(
        frame,
        "Q Quit   D Debug   S Screenshot   R Reset   F Fullscreen",
        (20, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )