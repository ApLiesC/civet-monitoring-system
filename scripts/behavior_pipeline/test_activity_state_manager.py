
from activity_state_manager import ActivityStateManager


manager = ActivityStateManager(
    transition_duration=3.0,
    unknown_tolerance=3.0,
    not_visible_tolerance=3.0,
)

# Simulate a civet moving, then resting, then disappearing.
samples = [
    ("active", 0.0),
    ("active", 1.0),
    ("active", 2.0),
    ("active", 3.0),

    ("inactive", 4.0),
    ("inactive", 5.0),
    ("inactive", 6.0),
    ("inactive", 7.0),

    (None, 8.0),  # Detection lost
    (None, 9.0),
    (None, 10.0),
    (None, 11.0),
    (None, 12.0),
]

for state, timestamp in samples:
    if state is None:
        result = manager.update(
            None,
            timestamp,
            unknown_reason="detection",
        )
    else:
        result = manager.update(state, timestamp)

    print(f"{timestamp:5.1f}s | {result}")

print("\nFinal records:")

for record in manager.finalize():
    print(record)
