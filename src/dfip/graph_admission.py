"""CPU-only bounded admission for an experimental resident CUDA graph set."""

from collections import Counter, deque


class AdaptiveAdmission:
    """Promote frequent shapes; retain captures without eviction or recapture.

    A capture attempt consumes a lifetime budget, including failed attempts.
    Explicit worker reconfiguration starts a new budget. Prediction values are
    never stored here. The caller must serialize mutable GPU state.
    """

    def __init__(self, max_captures=4, window=128, min_observations=32,
                 min_hits=8, min_fraction=.125):
        if not (all(type(v) is int for v in (max_captures, window, min_observations, min_hits))
                and 0 < max_captures and 0 < min_hits <= min_observations <= window
                and 0 < min_fraction <= 1):
            raise ValueError('Invalid adaptive graph admission limits')
        self.max_captures = max_captures
        self.window = window
        self.min_observations = min_observations
        self.min_hits = min_hits
        self.min_fraction = min_fraction
        self.recent = deque()
        self.counts = Counter()
        self.attempts = 0
        self.actions = Counter()
        self.frozen = False

    def select(self, key, *, cached, full):
        self.recent.append(key)
        self.counts[key] += 1
        if len(self.recent) > self.window:
            expired = self.recent.popleft()
            self.counts[expired] -= 1
            if not self.counts[expired]:
                del self.counts[expired]
        if cached:
            action = 'replay'
        elif full or self.attempts >= self.max_captures or self.frozen:
            action = 'native_budget'
        elif (len(self.recent) < self.min_observations
              or self.counts[key] < self.min_hits
              or self.counts[key] / len(self.recent) < self.min_fraction):
            action = 'native_cold'
        else:
            self.attempts += 1
            action = 'capture'
        self.actions[action] += 1
        return action

    def diagnostics(self):
        return {'policy': 'adaptive_pinned', 'max_capture_attempts': self.max_captures,
                'capture_attempts': self.attempts, 'window': self.window,
                'min_observations': self.min_observations, 'min_hits': self.min_hits,
                'min_fraction': self.min_fraction, 'recent_distinct_shapes': len(self.counts),
                'actions': dict(self.actions), 'admission_frozen': self.frozen}
