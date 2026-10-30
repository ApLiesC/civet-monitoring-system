"""fake data to test the dashboard before the model is ready."""
import csv
import os
import random
from datetime import date, timedelta

random.seed(7)
os.makedirs("data", exist_ok=True)
with open("data/events.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["date", "behavior", "start", "stop"])
    for i in range(7):
        d = date.today() - timedelta(days=6 - i)
        t = 0
        while t < 86000:
            b = random.choices(["inactive", "active", "feeding"], [45, 40, 15])[0]
            mins = {"feeding": 4, "active": 12, "inactive": 25}[b]
            length = int(mins * 60 * random.uniform(0.5, 1.5))
            w.writerow([d.isoformat(), b, t, min(t + length, 86400)])
            t += length + random.randint(0, 900)
print("wrote data/events.csv")
