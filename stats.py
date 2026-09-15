import time


class ReactionStats:
    def __init__(self):
        self.reset()

    def reset(self):
        now = time.time()

        self.start_time = now
        self.last_update = now

        self.current_expression = None

        self.counts = {
            "THINKING": 0,
            "SMILING": 0,
            "SHOCKED": 0
        }

        self.durations = {
            "THINKING": 0.0,
            "SMILING": 0.0,
            "SHOCKED": 0.0
        }

    def update(self, expression):
        now = time.time()

        # Add elapsed time to previous expression
        if self.current_expression is not None:
            elapsed = now - self.last_update

            self.durations[
                self.current_expression
            ] += elapsed

        # Count when expression changes
        if expression != self.current_expression:
            self.counts[expression] += 1

            self.current_expression = expression

        self.last_update = now

    def get_session_time(self):
        return time.time() - self.start_time