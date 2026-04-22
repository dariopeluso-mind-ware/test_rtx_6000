#!/usr/bin/env python3
"""
Prepara il dataset YOLO-OBB per il training di YOLO26 oriented bounding box.

Converte le annotazioni di segmentazione CVAT (poligoni) nel formato OBB
richiesto da Ultralytics:
    class_index x1 y1 x2 y2 x3 y3 x4 y4
    (coordinate normalizzate 0–1, 4 angoli del rettangolo ruotato)

Il dataset CVAT di partenza è l'export "YOLO Segmented" (yolo_seg.zip).
I file label OBB e det del CVAT risultavano vuoti — solo il formato segmentato
conserva le annotazioni effettive come poligoni.

NOTA: YOLO26 OBB vincola gli angoli all'intervallo [0°, 90°) come da docs
      ufficiali. Il rescaling con cv2.minAreaRect gestisce questo automaticamente.

Usage:
    python scripts/prepare_dataset_obb.py yolo_seg.zip ./data/tosano_obb \\
        --images-dir etichette_esempio/

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

Genera anche un file ZIP pronto per l'upload su Google Drive.

Refs:
    https://docs.ultralytics.com/tasks/obb/
    https://docs.ultralytics.com/datasets/obb/
"""

from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
# TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO = 0.10

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Classe unica del progetto Tosano
CLASS_NAMES = ["food_label"]


# ---------------------------------------------------------------------------
# Conversione segmentazione → OBB
# ---------------------------------------------------------------------------


def polygon_to_obb(coords: list[float]) -> list[float]:
    """
    Converte una sequenza di coordinate poligono normalizzate in 4 corner OBB.

    Per poligoni con 4 punti (quadrilateri) li restituisce direttamente.
    Per poligoni con più punti usa cv2.minAreaRect per trovare il rettangolo
    minimo ruotato, quindi estrae i 4 angoli con cv2.boxPoints.

    Args:
        coords: Lista piatta di coordinate normalizzate [x1, y1, x2, y2, ...].
                Deve avere lunghezza pari e ≥ 8 (almeno 4 punti).

    Returns:
        Lista di 8 float [x1, y1, x2, y2, x3, y3, x4, y4] normalizzati,
        corrispondenti ai 4 angoli del rettangolo orientato.

    Raises:
        ValueError: Se il numero di coordinate è dispari o < 8.
    """
    if len(coords) % 2 != 0:
        raise ValueError(
            f"Numero di coordinate dispari: {len(coords)}. Attesi coppie x,y."
        )
    if len(coords) < 8:
        raise ValueError(
            f"Almeno 4 punti richiesti (8 coordinate), trovati: {len(coords)}"
        )

    n_points = len(coords) // 2

    # Per quadrilateri a 4 punti: usa i punti direttamente come OBB
    # (CVAT esporta già quadrilateri orientati per le etichette rettangolari)
    if n_points == 4:
        return coords  # già nel formato x1 y1 x2 y2 x3 y3 x4 y4

    # Per poligoni con più punti: calcola il minAreaRect
    # Usa coordinate fittizie [0–10000] per preservare precisione float
    SCALE = 10_000.0
    points = np.array(
        [[coords[i] * SCALE, coords[i + 1] * SCALE] for i in range(0, len(coords), 2)],
        dtype=np.float32,
    )

    # cv2.minAreaRect restituisce ((cx, cy), (w, h), angle)
    rect = cv2.minAreaRect(points)

    # cv2.boxPoints restituisce i 4 angoli del rettangolo ruotato
    box = cv2.boxPoints(rect)  # shape (4, 2)

    # Rinormalizza dividendo per SCALE
    obb_coords: list[float] = []
    for pt in box:
        obb_coords.append(float(pt[0]) / SCALE)
        obb_coords.append(float(pt[1]) / SCALE)

    # Clamp [0, 1] per sicurezza (minAreaRect può produrre valori fuori range)
    obb_coords = [max(0.0, min(1.0, v)) for v in obb_coords]

    return obb_coords


def convert_seg_label_to_obb(label_content: str) -> str:
    """
    Converte il contenuto di un file label segmentazione in formato OBB.

    Formato input (YOLO segmentazione):
        class_index x1 y1 x2 y2 ... xN yN

    Formato output (YOLO OBB):
        class_index x1 y1 x2 y2 x3 y3 x4 y4

    Args:
        label_content: Contenuto testuale del file label YOLO segmentazione.

    Returns:
        Contenuto testuale nel formato YOLO OBB.

    Raises:
        ValueError: Se una riga ha formato non valido.
    """
    output_lines: list[str] = []

    for line_num, line in enumerate(label_content.strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue

        tokens = line.split()
        if len(tokens) < 9:  # class + almeno 4 punti = 9 token
            raise ValueError(
                f"Riga {line_num}: troppo pochi token ({len(tokens)}). "
                f"Attesi almeno 9 (class + 4 punti). Contenuto: '{line}'"
            )

        class_idx = tokens[0]
        coords = [float(t) for t in tokens[1:]]

        obb_coords = polygon_to_obb(coords)

        # Formato output: class_index x1 y1 x2 y2 x3 y3 x4 y4
        coord_str = " ".join(f"{c:.6f}" for c in obb_coords)
        output_lines.append(f"{class_idx} {coord_str}")

    return "\n".join(output_lines) + "\n" if output_lines else ""


# ---------------------------------------------------------------------------
# Estrazione CVAT export
# ---------------------------------------------------------------------------


def extract_cvat_seg_labels(
    zip_path: Path,
    work_dir: Path,
) -> tuple[Path, dict[str, str]]:
    """
    Estrae l'export CVAT segmentazione e restituisce i label come stringa.

    Args:
        zip_path: Path del file ZIP CVAT segmentazione.
        work_dir: Directory di lavoro temporanea.

    Returns:
        Tuple (extract_dir, {stem → contenuto_label_obb}).

    Raises:
        FileNotFoundError: Se non trova label nell'export.
    """
    extract_dir = work_dir / "cvat_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    print(f"📂 Estratto CVAT export in: {extract_dir}")

    # Cerca i file label nella struttura (con o senza subdirectory)
    label_files = list(extract_dir.rglob("*.txt"))
    # Esclude train.txt, obj.names, data.yaml, ecc.
    label_files = [
        f for f in label_files
        if f.name not in {"train.txt", "obj.names", "test.txt", "val.txt"}
        and "labels" in f.parts
    ]

    if not label_files:
        # Fallback: tutti i .txt che non sono manifest
        label_files = [
            f for f in extract_dir.rglob("*.txt")
            if f.name not in {"train.txt", "obj.names", "test.txt", "val.txt"}
        ]

    if not label_files:
        raise FileNotFoundError(
            f"Nessun file label trovato nell'export CVAT: {extract_dir}"
        )

    print(f"   Trovati {len(label_files)} file label")

    # Converti ogni label da segmentazione → OBB
    converted: dict[str, str] = {}
    errors: list[str] = []

    for lf in label_files:
        stem = lf.stem
        try:
            content = lf.read_text(encoding="utf-8")
            obb_content = convert_seg_label_to_obb(content)
            converted[stem] = obb_content
        except (ValueError, Exception) as e:
            errors.append(f"  ⚠️  {lf.name}: {e}")

    if errors:
        print(f"\n⚠️  {len(errors)} label con errori di conversione:")
        for err in errors[:10]:
            print(err)
        if len(errors) > 10:
            print(f"  ... e altri {len(errors) - 10}")

    print(f"✅ Label convertite in OBB: {len(converted)}")
    return extract_dir, converted


# ---------------------------------------------------------------------------
# Matching immagini ↔ label
# ---------------------------------------------------------------------------


def match_images_to_labels(
    images_dir: Path,
    labels: dict[str, str],
) -> list[tuple[Path, str]]:
    """
    Abbina le immagini nella directory ai label convertiti.

    Args:
        images_dir: Directory contenente le immagini originali.
        labels: Dizionario {stem → contenuto_label_obb}.

    Returns:
        Lista di tuple (image_path, label_content) per le coppie trovate.
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"Directory immagini non trovata: {images_dir}")

    matched: list[tuple[Path, str]] = []
    orphan_images: list[str] = []
    orphan_labels: set[str] = set(labels.keys())

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stem = img_path.stem
        if stem in labels:
            matched.append((img_path, labels[stem]))
            orphan_labels.discard(stem)
        else:
            orphan_images.append(img_path.name)

    if orphan_images:
        print(f"\n⚠️  {len(orphan_images)} immagini senza label (ignorate):")
        for name in orphan_images[:5]:
            print(f"   - {name}")
        if len(orphan_images) > 5:
            print(f"   ... e altre {len(orphan_images) - 5}")

    if orphan_labels:
        print(f"\n⚠️  {len(orphan_labels)} label senza immagine (ignorate):")
        for stem in list(orphan_labels)[:5]:
            print(f"   - {stem}.txt")
        if len(orphan_labels) > 5:
            print(f"   ... e altri {len(orphan_labels) - 5}")

    return matched


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def split_pairs(
    pairs: list[tuple[Path, str]],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> dict[str, list[tuple[Path, str]]]:
    """
    Divide le coppie (immagine, label) in train/val/test.

    Args:
        pairs: Lista di tuple (image_path, label_content).
        train_ratio: Frazione per il training set.
        val_ratio: Frazione per il validation set.
        seed: Seed per shuffle deterministico.

    Returns:
        Dizionario con chiavi 'train', 'val', 'test'.
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


# ---------------------------------------------------------------------------
# Dataset output
# ---------------------------------------------------------------------------


def create_dataset_structure(
    splits: dict[str, list[tuple[Path, str]]],
    output_dir: Path,
) -> None:
    """
    Crea la struttura di directory YOLO-OBB e scrive i file.

    Args:
        splits: Dizionario split → lista di (image_path, label_obb_content).
        output_dir: Directory di output.
    """
    for split_name, pairs in splits.items():
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, label_content in pairs:
            shutil.copy2(img_path, img_dir / img_path.name)
            lbl_file = lbl_dir / f"{img_path.stem}.txt"
            lbl_file.write_text(label_content, encoding="utf-8")

    print(f"\n✅ Dataset OBB creato in: {output_dir}")
    for split_name, pairs in splits.items():
        print(f"   {split_name}: {len(pairs)} immagini")


def write_data_yaml(output_dir: Path) -> Path:
    """
    Genera il file data.yaml compatibile YOLO-OBB.

    Args:
        output_dir: Directory root del dataset.

    Returns:
        Path del file data.yaml creato.
    """
    yaml_path = output_dir / "data.yaml"

    names_block = "\n".join(
        f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)
    )

    content = f"""# Dataset Tosano Food Labels - YOLO26 OBB
# Generato da prepare_dataset_obb.py
# Formato label: class_index x1 y1 x2 y2 x3 y3 x4 y4 (normalizzato 0-1)
# Ref: https://docs.ultralytics.com/tasks/obb/

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


def verify_obb_labels(output_dir: Path) -> bool:
    """
    Verifica che tutti i label abbiano il formato OBB corretto.

    Controlla che ogni riga abbia esattamente 9 token (class + 8 coordinate)
    e che le coordinate siano nell'intervallo [0, 1].

    Args:
        output_dir: Directory root del dataset.

    Returns:
        True se tutti i label sono validi, False altrimenti.
    """
    errors: list[str] = []
    total_labels = 0

    for lbl_file in sorted(output_dir.rglob("labels/**/*.txt")):
        for line_num, line in enumerate(
            lbl_file.read_text(encoding="utf-8").splitlines(), 1
        ):
            line = line.strip()
            if not line:
                continue
            total_labels += 1
            tokens = line.split()

            if len(tokens) != 9:
                errors.append(
                    f"{lbl_file.name}:{line_num}: "
                    f"attesi 9 token, trovati {len(tokens)}"
                )
                continue

            try:
                coords = [float(t) for t in tokens[1:]]
                out_of_range = [c for c in coords if not (0.0 <= c <= 1.0)]
                if out_of_range:
                    errors.append(
                        f"{lbl_file.name}:{line_num}: "
                        f"coordinate fuori [0,1]: {out_of_range}"
                    )
            except ValueError as e:
                errors.append(f"{lbl_file.name}:{line_num}: {e}")

    if errors:
        print(f"\n❌ {len(errors)} errori di validazione label:")
        for err in errors[:10]:
            print(f"   {err}")
        if len(errors) > 10:
            print(f"   ... e altri {len(errors) - 10}")
        return False

    print(f"\n✅ Validazione label: {total_labels} annotation OBB verificate")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point principale."""
    parser = argparse.ArgumentParser(
        description=(
            "Converte il CVAT segmentation export in dataset YOLO-OBB\n"
            "per il training di YOLO26 oriented bounding box.\n\n"
            "Ref: https://docs.ultralytics.com/tasks/obb/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "cvat_seg_zip",
        type=Path,
        help="Path del file ZIP CVAT in formato YOLO Segmented (es. yolo_seg.zip)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory di output per il dataset OBB strutturato",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("etichette_esempio"),
        help=(
            "Directory contenente le immagini originali "
            "(default: etichette_esempio/)"
        ),
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

    # Validazioni input
    if not args.cvat_seg_zip.exists():
        raise FileNotFoundError(f"File ZIP non trovato: {args.cvat_seg_zip}")

    if not args.images_dir.exists():
        raise FileNotFoundError(
            f"Directory immagini non trovata: {args.images_dir}\n"
            "Usa --images-dir per specificare il path corretto."
        )

    ratios_sum = args.train_ratio + args.val_ratio
    if ratios_sum >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio deve essere < 1.0, "
            f"ottenuto {ratios_sum:.2f}"
        )
    test_ratio = 1.0 - ratios_sum

    print("=" * 65)
    print("📋 PREPARAZIONE DATASET TOSANO OBB - YOLO26")
    print("=" * 65)
    print(f"Input ZIP:     {args.cvat_seg_zip}")
    print(f"Immagini:      {args.images_dir}")
    print(f"Output:        {args.output_dir}")
    print(f"Split:         train={args.train_ratio:.0%} / "
          f"val={args.val_ratio:.0%} / test={test_ratio:.0%}")
    print(f"Seed:          {args.seed}")
    print()

    # 1. Estrai e converti label da segmentazione → OBB
    work_dir = args.output_dir.parent / f".{args.output_dir.name}_work"
    _, labels_obb = extract_cvat_seg_labels(args.cvat_seg_zip, work_dir)

    if not labels_obb:
        raise ValueError("Nessun label OBB convertito! Controlla il file ZIP.")

    print(f"\n🔄 {len(labels_obb)} label convertiti da segmentazione a OBB")

    # 2. Abbina immagini ai label
    matched = match_images_to_labels(args.images_dir, labels_obb)
    if not matched:
        raise ValueError(
            "Nessuna coppia immagine-label trovata!\n"
            f"Immagini in: {args.images_dir}\n"
            f"Label stems: {sorted(labels_obb.keys())[:5]}..."
        )
    print(f"🔍 {len(matched)} coppie immagine-label abbinate")

    # 3. Split
    splits = split_pairs(
        matched,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # 4. Crea struttura dataset
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    create_dataset_structure(splits, args.output_dir)

    # 5. Genera data.yaml
    write_data_yaml(args.output_dir)

    # 6. Validazione label OBB
    is_valid = verify_obb_labels(args.output_dir)
    if not is_valid:
        print("\n⚠️  Attenzione: alcuni label non hanno il formato OBB corretto.")

    # 7. Crea ZIP per Google Drive
    if not args.no_zip:
        create_zip(args.output_dir)

    # 8. Pulizia directory di lavoro
    if work_dir.exists():
        shutil.rmtree(work_dir)

    print(f"\n{'=' * 65}")
    print("✅ COMPLETATO!")
    print(f"{'=' * 65}")
    test_ratio_pct = f"{test_ratio:.0%}"
    print(f"\nDataset OBB pronto con split "
          f"{args.train_ratio:.0%}/{args.val_ratio:.0%}/{test_ratio_pct}")
    print(f"\nProssimi passi:")
    if not args.no_zip:
        print(f"  1. Carica {args.output_dir.name}.zip su Google Drive")
    print(f"  2. Apri train_yolo26_obb_tosano.py su Google Colab")
    print(f"  3. Aggiorna DRIVE_ZIP_PATH con il path su Drive")
    print(f"  4. Seleziona GPU Runtime (T4/L4/A100)")
    print(f"  5. Avvia il training!")


if __name__ == "__main__":
    main()
