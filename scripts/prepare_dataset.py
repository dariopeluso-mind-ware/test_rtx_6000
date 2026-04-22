#!/usr/bin/env python3
"""
Prepara il dataset YOLO per il training di YOLO26 detection.

Prende l'export CVAT in formato "YOLO 1.1" (zip) e lo ristruttura
nella directory standard YOLO con split train/val/test.

Usage:
    python prepare_dataset.py /path/to/cvat_export.zip /path/to/output_dir

L'output sarà:
    output_dir/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml

Lo script genera anche un file ZIP pronto per l'upload su Google Drive.
"""

from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path

# Proporzioni di split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# Seed fisso per riproducibilità
RANDOM_SEED = 42

# Estensioni immagine supportate
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Classe unica per il progetto Tosano
CLASS_NAMES = ["food_label"]


def find_image_label_pairs(
    images_dir: Path,
    labels_dir: Path,
) -> list[tuple[Path, Path]]:
    """
    Trova le coppie (immagine, label) corrispondenti.

    Args:
        images_dir: Directory contenente le immagini.
        labels_dir: Directory contenente i file .txt YOLO.

    Returns:
        Lista di tuple (image_path, label_path) per i file che hanno
        sia immagine che label.

    Raises:
        FileNotFoundError: Se le directory non esistono.
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"Directory immagini non trovata: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Directory label non trovata: {labels_dir}")

    pairs: list[tuple[Path, Path]] = []
    orphan_images: list[str] = []

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            orphan_images.append(img_path.name)

    if orphan_images:
        print(f"⚠️  {len(orphan_images)} immagini senza label (saranno ignorate):")
        for name in orphan_images[:10]:
            print(f"    - {name}")
        if len(orphan_images) > 10:
            print(f"    ... e altre {len(orphan_images) - 10}")

    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> dict[str, list[tuple[Path, Path]]]:
    """
    Divide le coppie in train/val/test con shuffle deterministico.

    Args:
        pairs: Lista di coppie (image, label).
        train_ratio: Frazione per il training set.
        val_ratio: Frazione per il validation set.
        seed: Seed per il random shuffle.

    Returns:
        Dizionario con chiavi 'train', 'val', 'test' e liste di coppie.
    """
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def create_dataset_structure(
    splits: dict[str, list[tuple[Path, Path]]],
    output_dir: Path,
) -> None:
    """
    Crea la struttura di directory YOLO e copia i file.

    Args:
        splits: Dizionario con split 'train', 'val', 'test'.
        output_dir: Directory di output.
    """
    for split_name, pairs in splits.items():
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, img_dir / img_path.name)
            shutil.copy2(lbl_path, lbl_dir / lbl_path.name)

    print(f"\n✅ Dataset creato in: {output_dir}")
    for split_name, pairs in splits.items():
        print(f"   {split_name}: {len(pairs)} immagini")


def write_data_yaml(output_dir: Path) -> Path:
    """
    Genera il file data.yaml per Ultralytics YOLO.

    Args:
        output_dir: Directory root del dataset.

    Returns:
        Path del file data.yaml creato.
    """
    yaml_path = output_dir / "data.yaml"

    # Usiamo path relativo (verrà aggiornato nel notebook Colab)
    names_block = "\n".join(
        f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)
    )

    content = f"""# Dataset Tosano Food Labels - YOLO26 Detection
# Generato da prepare_dataset.py

path: {output_dir.resolve()}
train: images/train
val: images/val
test: images/test

names:
{names_block}
"""

    yaml_path.write_text(content, encoding="utf-8")
    print(f"📄 data.yaml creato: {yaml_path}")
    return yaml_path


def create_zip(output_dir: Path) -> Path:
    """
    Crea un file ZIP del dataset per upload su Google Drive.

    Args:
        output_dir: Directory del dataset.

    Returns:
        Path del file ZIP creato.
    """
    zip_path = output_dir.parent / f"{output_dir.name}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(output_dir.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(output_dir)
                zf.write(file_path, arcname)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"📦 ZIP creato: {zip_path} ({size_mb:.1f} MB)")
    return zip_path


def extract_cvat_export(zip_path: Path, work_dir: Path) -> tuple[Path, Path]:
    """
    Estrae l'export CVAT e individua le directory images/labels.

    CVAT YOLO 1.1 export produce tipicamente:
    - obj_train_data/  (immagini + label nella stessa cartella)
    - obj.names
    - obj.data
    oppure:
    - images/ e labels/ separati

    Args:
        zip_path: Path del file ZIP CVAT.
        work_dir: Directory di lavoro per l'estrazione.

    Returns:
        Tuple (images_dir, labels_dir).

    Raises:
        FileNotFoundError: Se non trova le immagini/label nell'export.
    """
    extract_dir = work_dir / "cvat_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    print(f"📂 Estratto CVAT export in: {extract_dir}")

    # Caso 1: CVAT YOLO 1.1 con obj_train_data/ (immagini e label insieme)
    obj_train = extract_dir / "obj_train_data"
    if obj_train.exists():
        print("   Formato: CVAT YOLO 1.1 (obj_train_data/)")
        # Separa immagini e label in cartelle distinte
        images_dir = work_dir / "sorted_images"
        labels_dir = work_dir / "sorted_labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for f in obj_train.iterdir():
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                shutil.copy2(f, images_dir / f.name)
            elif f.suffix == ".txt":
                shutil.copy2(f, labels_dir / f.name)

        return images_dir, labels_dir

    # Caso 2: struttura images/ + labels/ già separata
    for sub in [extract_dir] + list(extract_dir.iterdir()):
        if not sub.is_dir():
            continue
        imgs = sub / "images"
        lbls = sub / "labels"
        if imgs.exists() and lbls.exists():
            print("   Formato: images/ + labels/ separati")
            return imgs, lbls

    # Caso 3: tutto nella root dell'archivio
    has_images = any(
        f.suffix.lower() in IMAGE_EXTENSIONS for f in extract_dir.iterdir()
    )
    has_labels = any(f.suffix == ".txt" for f in extract_dir.iterdir())
    if has_images and has_labels:
        print("   Formato: file misti nella root")
        images_dir = work_dir / "sorted_images"
        labels_dir = work_dir / "sorted_labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for f in extract_dir.iterdir():
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                shutil.copy2(f, images_dir / f.name)
            elif f.suffix == ".txt" and f.name != "obj.names":
                shutil.copy2(f, labels_dir / f.name)

        return images_dir, labels_dir

    raise FileNotFoundError(
        f"Non riesco a trovare immagini e label nell'export CVAT: {extract_dir}\n"
        f"Contenuto: {[f.name for f in extract_dir.iterdir()]}"
    )


def main() -> None:
    """Entry point principale."""
    parser = argparse.ArgumentParser(
        description="Prepara dataset YOLO da export CVAT per training YOLO26.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "cvat_zip",
        type=Path,
        help="Path del file ZIP esportato da CVAT (formato YOLO 1.1)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory di output per il dataset strutturato",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help=f"Proporzione training set (default: {TRAIN_RATIO})",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help=f"Proporzione validation set (default: {VAL_RATIO})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Seed per shuffle deterministico (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Non creare il file ZIP finale",
    )

    args = parser.parse_args()

    if not args.cvat_zip.exists():
        raise FileNotFoundError(f"File non trovato: {args.cvat_zip}")

    ratios_sum = args.train_ratio + args.val_ratio
    if ratios_sum >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio deve essere < 1.0, "
            f"ottenuto {ratios_sum:.2f}"
        )
    test_ratio = 1.0 - ratios_sum

    print("=" * 60)
    print("📋 PREPARAZIONE DATASET TOSANO FOOD LABELS")
    print("=" * 60)
    print(f"Input:  {args.cvat_zip}")
    print(f"Output: {args.output_dir}")
    print(f"Split:  train={args.train_ratio:.0%} / "
          f"val={args.val_ratio:.0%} / test={test_ratio:.0%}")
    print(f"Seed:   {args.seed}")
    print()

    # 1. Estrai export CVAT
    work_dir = args.output_dir.parent / f".{args.output_dir.name}_work"
    images_dir, labels_dir = extract_cvat_export(args.cvat_zip, work_dir)

    # 2. Trova coppie immagine-label
    pairs = find_image_label_pairs(images_dir, labels_dir)
    if not pairs:
        raise ValueError("Nessuna coppia immagine-label trovata!")
    print(f"\n🔍 Trovate {len(pairs)} coppie immagine-label")

    # 3. Split
    splits = split_pairs(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # 4. Crea struttura e copia file
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    create_dataset_structure(splits, args.output_dir)

    # 5. Genera data.yaml
    write_data_yaml(args.output_dir)

    # 6. Crea ZIP per Google Drive
    if not args.no_zip:
        create_zip(args.output_dir)

    # 7. Pulizia directory di lavoro
    if work_dir.exists():
        shutil.rmtree(work_dir)

    print(f"\n{'=' * 60}")
    print("✅ COMPLETATO!")
    print(f"{'=' * 60}")
    print(f"\nProssimi passi:")
    print(f"  1. Carica {args.output_dir.name}.zip su Google Drive")
    print(f"  2. Apri il notebook Colab train_yolo26_det_tosano.py")
    print(f"  3. Aggiorna DRIVE_ZIP_PATH con il path su Drive")
    print(f"  4. Avvia il training!")


if __name__ == "__main__":
    main()
