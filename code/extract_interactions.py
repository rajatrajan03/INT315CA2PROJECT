
from pathlib import Path
import csv

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NETWORKS = ["ICS", "CD", "PS"]


def extract_network(filename, network_name):
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    students = []
    interactions = []
    section = None

    with file_path.open("r", encoding="utf-8-sig", errors="replace") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line or line.startswith("%"):
                continue

            lower_line = line.lower()

            if lower_line.startswith("*vertices"):
                section = "vertices"
                continue
            elif lower_line.startswith("*arcs"):
                section = "arcs"
                continue
            elif lower_line.startswith("*edges"):
                section = "edges"
                continue
            elif line.startswith("*"):
                section = None
                continue

            parts = line.split()

            if section == "vertices":
                # Vertex lines usually start with an ID and a label.
                student_id = parts[0]
                label = parts[1].strip('"') if len(parts) > 1 else student_id

                students.append({
                    "network": network_name,
                    "student_id": student_id,
                    "label": label
                })

            elif section in ("arcs", "edges"):
                if len(parts) < 2:
                    continue

                source = parts[0]
                target = parts[1]

                # Preserve the original weight if supplied.
                weight = parts[2] if len(parts) >= 3 else "1"

                interactions.append({
                    "network": network_name,
                    "source": source,
                    "target": target,
                    "weight": weight,
                    "interaction_type": section
                })

    return students, interactions


all_students = []
all_interactions = []

for network in NETWORKS:
    filename = f"{network}_RAW.net"
    students, interactions = extract_network(filename, network)

    all_students.extend(students)
    all_interactions.extend(interactions)

    print(f"\n{network} dataset")
    print(f"Students: {len(students)}")
    print(f"Interactions: {len(interactions)}")

# Save student records
student_file = OUTPUT_DIR / "students.csv"

with student_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["network", "student_id", "label"]
    )
    writer.writeheader()
    writer.writerows(all_students)

# Save interaction records
interaction_file = OUTPUT_DIR / "interactions.csv"

with interaction_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "network", "source", "target",
            "weight", "interaction_type"
        ]
    )
    writer.writeheader()
    writer.writerows(all_interactions)

print("\nExtraction completed successfully.")
print(f"Student CSV: {student_file}")
print(f"Interaction CSV: {interaction_file}")
