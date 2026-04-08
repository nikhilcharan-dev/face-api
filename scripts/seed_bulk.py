#!/usr/bin/env python3
"""
Bulk face enrollment seeder.

Reads every image in a directory, extracts a 512-D ArcFace embedding,
and upserts it directly into MongoDB — no HTTP round-trips, no FAISS files.

Image naming convention
-----------------------
  {employeeId}.jpg            →  employeeId = filename stem
  {employeeId}_{tag}.jpg      →  employeeId = part before first underscore

Supported formats: .jpg .jpeg .png .webp .bmp

Usage
-----
  python scripts/seed_bulk.py \\
      --dir        image_dump \\
      --org-id     <MongoDB organization ObjectId> \\
      --mongodb-uri mongodb+srv://... \\
      [--db        workping] \\
      [--dry-run]
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# allow running from the project root or the scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402
from embedding import load_face_app, get_face_embedding_from_bytes  # noqa: E402

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def employee_id_from_path(p: Path) -> str:
    """
    Derive employeeId from the file stem.
    'EMP001_front.jpg'  →  'EMP001'
    'EMP001.jpg'        →  'EMP001'
    """
    stem = p.stem
    return stem.split("_")[0]


async def run(image_dir: str, org_id: str, mongodb_uri: str, db_name: str, dry_run: bool):
    client = AsyncIOMotorClient(mongodb_uri)
    col = client[db_name]["faceembeddings"]

    print("Loading InsightFace model...")
    load_face_app()
    print("Model ready.\n")

    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        print(f"[ERROR] Directory not found: {image_dir}")
        sys.exit(1)

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not images:
        print(f"[WARN]  No supported images found in {image_dir}")
        return

    print(f"Found {len(images)} image(s).  org_id={org_id}  dry_run={dry_run}\n")

    ok = fail = skipped = 0
    for img_path in images:
        emp_id = employee_id_from_path(img_path)
        try:
            image_bytes = img_path.read_bytes()
            emb = get_face_embedding_from_bytes(image_bytes)

            if dry_run:
                print(f"[DRY]  {emp_id}  ({img_path.name})")
                skipped += 1
                continue

            await col.update_one(
                {"employee_id": emp_id},
                {"$set": {
                    "employee_id": emp_id,
                    "organization_id": org_id,
                    "embedding": emb.tolist(),
                }},
                upsert=True,
            )
            print(f"[OK]   {emp_id}  ({img_path.name})")
            ok += 1
        except Exception as exc:
            print(f"[FAIL] {emp_id}  ({img_path.name}): {exc}")
            fail += 1

    client.close()

    print(f"\n--- Summary ---")
    if dry_run:
        print(f"  Would enroll: {skipped}")
    else:
        print(f"  Enrolled : {ok}")
        print(f"  Failed   : {fail}")


def main():
    parser = argparse.ArgumentParser(
        description="Bulk enroll face images into MongoDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dir", default="image_dump",
        help="Directory containing face images (default: image_dump)",
    )
    parser.add_argument(
        "--org-id", required=True,
        help="MongoDB Organization ObjectId string",
    )
    parser.add_argument(
        "--mongodb-uri",
        default=os.getenv("MONGODB_URI"),
        help="MongoDB connection URI (overrides MONGODB_URI env var)",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("MONGODB_DB", "workping"),
        help="MongoDB database name (default: workping)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse images and extract embeddings but do NOT write to MongoDB",
    )
    args = parser.parse_args()

    if not args.mongodb_uri:
        parser.error("--mongodb-uri is required (or set MONGODB_URI env var)")

    asyncio.run(run(args.dir, args.org_id, args.mongodb_uri, args.db, args.dry_run))


if __name__ == "__main__":
    main()
