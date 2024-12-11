import os, inspect, csv

# charuco.py を使ってモニタの姿勢を測定しておく。
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
fp = os.path.join(currentdir, "charuco_origin.csv")
if os.path.exists(fp):
    rows = []
    with open(fp, newline="") as f:
        csvreader = csv.reader(f, delimiter=",")
        for row in csvreader:
            rows.append(row)
    if len(rows):
        FIX_Z = float(rows[0][2])
    else:
        FIX_Z = 0.2
else:
    FIX_Z = 0.2
