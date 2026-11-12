
from eating_event_tracker import EatingEventTracker


tracker = EatingEventTracker(
    minimum_duration=5.0,
    gap_tolerance=3.0,
)

samples = [
    # Short eating episode: should be discarded.
    ("eating", 0.0),
    ("eating", 1.0),
    ("eating", 2.0),
    ("resting", 3.0),
    ("resting", 4.0),
    ("resting", 5.0),
    ("resting", 6.0),

    # Eating episode with a brief interruption.
    ("eating", 10.0),
    ("eating", 11.0),
    ("eating", 12.0),
    ("unknown", 13.0),
    ("eating", 14.0),
    ("eating", 15.0),
    ("eating", 16.0),

    # Eating ends.
    ("resting", 17.0),
    ("resting", 18.0),
    ("resting", 19.0),
    ("resting", 20.0),
    ("resting", 21.0),
]

for behavior, timestamp in samples:
    result = tracker.update(behavior, timestamp)

    print(
        f"{timestamp:5.1f}s | "
        f"{behavior:8s} | "
        f"confirmed={result['confirmed']} | "
        f"new={result['newly_confirmed']}"
    )

print("\nConfirmed eating episodes:")

events = tracker.finalize()

for event in events:
    print(event)
