
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

files = [
    "ICS_RAW.net",
    "CD_RAW.net",
    "PS_RAW.net"
]

print("ONLINE LEARNING PLATFORM INTERACTION")
print("-" * 45)

for filename in files:
    file_path = DATA_DIR / filename

    if not file_path.exists():
        print(f"\nNOT FOUND: {filename}")
        continue

    with file_path.open("r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()

    vertices = sum(
        1 for line in lines
        if line.strip().lower().startswith("*vertices")
    )
    arcs = sum(
        1 for line in lines
        if line.strip().lower().startswith("*arcs")
    )
    edges = sum(
        1 for line in lines
        if line.strip().lower().startswith("*edges")
    )

    print(f"\nFile: {filename}")
    print(f"Total lines: {len(lines)}")
    print(f"Vertices section found: {vertices > 0}")
    print(f"Arcs section found: {arcs > 0}")
    print(f"Edges section found: {edges > 0}")

print("\nDataset check complete.")
