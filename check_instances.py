"""
To check validity of instances.jsonl
"""

import json

bad = 0

with open("instances.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        try:
            r = json.loads(line)

            F = int(r["F"])
            X = int(r["X"])
            Y = int(r["Y"])
            x_bits = int(r["meta"]["x_bits"])
            y_bits = int(r["meta"]["y_bits"])

            if F != X * Y:
                print(f"Line {i}: F != X * Y")
                bad += 1
                continue

            if X >= (1 << x_bits):
                print(f"Line {i}: X >= 2**x_bits  (X={X}, x_bits={x_bits})")
                bad += 1
                continue

            if Y >= (1 << y_bits):
                print(f"Line {i}: Y >= 2**y_bits  (Y={Y}, y_bits={y_bits})")
                bad += 1
                continue

            if X < (1 << (x_bits - 1)):
                print(f"Line {i}: X uses fewer than x_bits (X={X}, x_bits={x_bits})")
                bad += 1
                continue

            if Y < (1 << (y_bits - 1)):
                print(f"Line {i}: Y uses fewer than y_bits (Y={Y}, y_bits={y_bits})")
                bad += 1
                continue

        except Exception as e:
            print(f"Line {i}: {e}")
            bad += 1

print("Bad lines:", bad)
