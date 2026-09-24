"""
Chia tập train/val (và test tuỳ chọn) MỘT LẦN DUY NHẤT rồi ghi lại (seed cố
định) để mọi lần train sau đều dùng đúng cùng một tập train/val - tránh lỗi
"mỗi lần train một tập val khác nhau" do np.random.shuffle không seed.

Thư mục out_dir chỉ chứa dữ liệu ĐÃ COPY (không phải data/ gốc), nên xoá/ghi
đè out_dir là an toàn, KHÔNG đụng tới data/ gốc.
"""

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _find_images(images_dir: Path) -> List[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _load_class_names(classes_file: str) -> List[str]:
    path = Path(classes_file)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file classes: {classes_file}")
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if not names:
        raise ValueError(f"File classes rỗng: {classes_file}")
    return names


def _sha1_of_names(names: List[str]) -> str:
    joined = "\n".join(sorted(names))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _copy_pair(img_file: Path, labels_dir: Path, dst_images: Path, dst_labels: Path,
                allow_missing_label: bool = False) -> None:
    shutil.copy(img_file, dst_images / img_file.name)
    txt_file = labels_dir / f"{img_file.stem}.txt"
    dst_txt = dst_labels / f"{img_file.stem}.txt"
    if txt_file.exists():
        shutil.copy(txt_file, dst_txt)
    elif allow_missing_label:
        # Ảnh nền không có nhãn thủ công -> tạo file rỗng (0 box) để YOLO
        # coi đây là ảnh nền hợp lệ thay vì bỏ sót.
        dst_txt.touch()
    else:
        raise FileNotFoundError(f"Thiếu file nhãn cho ảnh {img_file}")


def _count_instances(files: List[Path], labels_dir: Path, num_classes: int) -> List[int]:
    counts = [0] * num_classes
    for img_file in files:
        txt_file = labels_dir / f"{img_file.stem}.txt"
        if not txt_file.exists():
            continue
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if 0 <= cls_id < num_classes:
                    counts[cls_id] += 1
    return counts


def _print_split_stats(split_name: str, files: List[Path], labels_dir: Path, class_names: List[str]) -> None:
    counts = _count_instances(files, labels_dir, len(class_names))
    print(f"\n📊 {split_name}: {len(files)} ảnh")
    zero_classes = []
    for name, n in zip(class_names, counts):
        print(f"   {name}: {n}")
        if n == 0:
            zero_classes.append(name)
    if zero_classes and split_name.lower() != "train":
        print(f"   ⚠️  Lớp không có instance nào trong {split_name}: {', '.join(zero_classes)}")


def build_dataset_once(
    images_dir: str,
    labels_dir: str,
    classes_file: str,
    out_dir: str = "yolo_dataset_v2",
    val_ratio: float = 0.15,
    seed: int = 0,
    test_dir: Optional[str] = None,
) -> Dict:
    """Tạo (hoặc tái sử dụng) bộ chia train/val/test cố định trong out_dir.

    Nếu out_dir/split.json đã tồn tại, dùng lại đúng danh sách file đã chia
    trước đó (không random lại) để đảm bảo mọi lần train dùng cùng 1 tập val.

    Returns:
        dict gồm: out_dir, data_yaml (đường dẫn), split_json (đường dẫn),
        train_files/val_files/test_files (tên file), num_classes.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)
    class_names = _load_class_names(classes_file)

    split_json_path = out_dir / "split.json"
    data_yaml_path = out_dir / "data.yaml"

    for split in ["train", "val"]:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    has_test = test_dir is not None
    if has_test:
        (out_dir / "test" / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "test" / "labels").mkdir(parents=True, exist_ok=True)

    if split_json_path.exists():
        print(f"♻️  Dùng lại split đã có: {split_json_path}")
        with open(split_json_path, "r", encoding="utf-8") as f:
            split_info = json.load(f)
        train_names = split_info["train_files"]
        val_names = split_info["val_files"]
        test_names = split_info.get("test_files", [])
    else:
        print("🔀 Chia dữ liệu train/val lần đầu (seed cố định)...")
        image_files = [
            img for img in _find_images(images_dir)
            if (labels_dir / f"{img.stem}.txt").exists()
        ]
        if not image_files:
            raise FileNotFoundError(
                "Không tìm thấy cặp ảnh/label hợp lệ. "
                "Kiểm tra lại images_dir và labels_dir."
            )

        rng = random.Random(seed)
        shuffled = image_files[:]
        rng.shuffle(shuffled)

        if len(shuffled) < 5:
            train_files, val_files = shuffled, []
        else:
            split_idx = int((1 - val_ratio) * len(shuffled))
            train_files = shuffled[:split_idx]
            val_files = shuffled[split_idx:]

        for img_file in train_files:
            _copy_pair(img_file, labels_dir, out_dir / "train" / "images", out_dir / "train" / "labels")
        for img_file in val_files:
            _copy_pair(img_file, labels_dir, out_dir / "val" / "images", out_dir / "val" / "labels")

        train_names = [p.name for p in train_files]
        val_names = [p.name for p in val_files]
        test_names = []

        if has_test:
            test_images_dir = Path(test_dir)
            test_files = _find_images(test_images_dir)
            for img_file in test_files:
                # Test set cho phép nhãn rỗng (ảnh nền) - không bắt buộc phải
                # có box, khác với train/val vốn chỉ lấy ảnh đã gán nhãn.
                _copy_pair(img_file, test_images_dir, out_dir / "test" / "images", out_dir / "test" / "labels",
                           allow_missing_label=True)
            test_names = [p.name for p in test_files]

        split_info = {
            "seed": seed,
            "val_ratio": val_ratio,
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "test_dir": str(test_dir) if has_test else None,
            "train_files": train_names,
            "val_files": val_names,
            "test_files": test_names,
            "sha1_train": _sha1_of_names(train_names),
            "sha1_val": _sha1_of_names(val_names),
            "sha1_test": _sha1_of_names(test_names) if test_names else None,
        }
        with open(split_json_path, "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)

    data_yaml = {
        "path": str(out_dir.absolute()),
        "train": "train/images",
        "val": "val/images" if val_names else "train/images",
        "nc": len(class_names),
        "names": class_names,
    }
    if test_names:
        data_yaml["test"] = "test/images"

    import yaml
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f)

    train_files = [out_dir / "train" / "images" / n for n in train_names]
    val_files = [out_dir / "val" / "images" / n for n in val_names]
    _print_split_stats("train", train_files, out_dir / "train" / "labels", class_names)
    _print_split_stats("val", val_files, out_dir / "val" / "labels", class_names)
    if test_names:
        test_files = [out_dir / "test" / "images" / n for n in test_names]
        _print_split_stats("test", test_files, out_dir / "test" / "labels", class_names)

    return {
        "out_dir": str(out_dir),
        "data_yaml": str(data_yaml_path),
        "split_json": str(split_json_path),
        "train_files": train_names,
        "val_files": val_names,
        "test_files": test_names,
        "num_classes": len(class_names),
    }


if __name__ == "__main__":
    info = build_dataset_once(
        images_dir="data/img",
        labels_dir="data/labels_yolo",
        classes_file="data/classes.txt",
    )
    print("\n✅ data.yaml:", info["data_yaml"])
