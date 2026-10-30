"""Small Flask server: serves the dashboard and the events as JSON.
Run from the project root:  python api/app.py   ->  http://localhost:5000
"""
import csv
import os
from flask import Flask, jsonify, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS_CSV = os.path.join(ROOT, "data", "events.csv")
app = Flask(__name__, static_folder=os.path.join(ROOT, "frontend"), static_url_path="")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/events")
def events():
    if not os.path.exists(EVENTS_CSV):
        return jsonify([])
    with open(EVENTS_CSV, newline="") as f:
        rows = [
            {"date": r["date"], "behavior": r["behavior"],
             "start": int(r["start"]), "stop": int(r["stop"])}
            for r in csv.DictReader(f)
        ]
    return jsonify(rows)


if __name__ == "__main__":
    app.run(debug=True)
