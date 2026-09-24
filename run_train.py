"""
Chạy 1 cấu hình train YOLO cho mục tiêu triển khai thực tế (phát hiện
RF/drone trên ảnh phổ đồ). Mỗi lần chạy script này tạo MỚI một
AutoLabelingSystem (không tái sử dụng model đã load) và ghi lại đầy đủ
thông tin cấu hình vào <save_dir>/experiment.json để so sánh giữa các lần
chạy sau này.

Ví dụ (xem thêm README ở cuối file cho 3 cấu hình A/B/C cần chạy):
    python3 run_train.py --model yolo26n.pt --epochs 150 --patience 50 \\
        --batch 16 --name A_yolo26n_notile

Cấu hình dùng chung 1 bộ split (yolo_dataset_v2/split.json, seed cố định) -
xem dataset_split.py.
"""

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import torch
import ultralytics

from autoLabel import AutoLabelingSystem
from augmentations import build_custom_augmentations

# Tham số augmentation/hyperparameter chuẩn cho ảnh phổ đồ (xem augmentations.py
# và giải thích trong autoLabel.py main()) - dùng chung cho mọi cấu hình A/B/C
# để so sánh công bằng, chỉ khác nhau ở model/tile.
TRAIN_HYPERPARAMS = dict(
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.2,
    degrees=0.0,
    shear=0.0,
    perspective=0.0,
    translate=0.0,   # tắt dịch dọc (y) mặc định của YOLO - xem augmentations.py cho dịch x riêng
    scale=0.05,
    flipud=0.5,
    fliplr=0.0,
    mosaic=0.0,
    mixup=0.0,
    erasing=0.0,
)


class _Tee(io.TextIOBase):
    """Ghi đồng thời ra stdout thật + buffer, để vừa xem log trực tiếp vừa
    bắt lại dòng "Transferred X/Y items from pretrained weights" cho experiment.json."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description="Train 1 cấu hình YOLO cho triển khai phát hiện RF/drone")
    p.add_argument("--model", required=True, help="model_path (.pt hoặc .yaml kiến trúc)")
    p.add_argument("--pretrained", default=None, help="pretrained weights .pt nếu --model là .yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--tile", action="store_true", help="bật tiling train (tile_train_data=True)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", required=True, help="tên run (vd A_yolo26n_notile)")
    p.add_argument("--project", default="model_deploy")
    p.add_argument("--fraction", type=float, default=1.0, help="tỉ lệ dữ liệu train dùng (smoke test dùng vd 0.05)")
    p.add_argument("--all-data", action="store_true",
                    help="train trên toàn bộ ảnh (không tách val) - dùng out_dir riêng "
                         "(<out_dir>_alldata) để không đụng split chuẩn có val")
    p.add_argument("--out-dir", default="yolo_dataset_v2")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-dir", default=None)
    p.add_argument("--images-dir", default="data/img")
    p.add_argument("--labels-dir", default="data/labels_yolo")
    p.add_argument("--classes-file", default="data/classes.txt")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = args.out_dir + "_alldata" if args.all_data else args.out_dir
    val_ratio = 0.0 if args.all_data else args.val_ratio

    buf = io.StringIO()
    tee = _Tee(sys.stdout, buf)
    with contextlib.redirect_stdout(tee):
        system = AutoLabelingSystem(
            model_path=args.model,
            pretrained_weights=args.pretrained,
            conf_threshold=0.5,
        )
    load_log = buf.getvalue()
    transferred_line = next((l.strip() for l in load_log.splitlines() if "Transferred" in l), None)

    train_kwargs = dict(TRAIN_HYPERPARAMS)
    train_kwargs["augmentations"] = build_custom_augmentations(images_dir=args.images_dir)
    if args.fraction != 1.0:
        train_kwargs["fraction"] = args.fraction

    try:
        results = system.train_on_labeled_data(
            labeled_images_dir=args.images_dir,
            labeled_annotations_dir=args.labels_dir,
            classes_file=args.classes_file,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            patience=args.patience,
            project=args.project,
            name=args.name,
            tile_train_data=args.tile,
            out_dir=out_dir,
            val_ratio=val_ratio,
            seed=args.seed,
            test_dir=args.test_dir,
            **train_kwargs,
        )
    except torch.cuda.OutOfMemoryError as exc:
        print(f"\n❌ OOM khi train '{args.name}' (batch={args.batch}). "
              f"Không tự động giảm batch - báo lại để quyết định thủ công.\n{exc}")
        raise
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(f"\n❌ OOM khi train '{args.name}' (batch={args.batch}). "
                  f"Không tự động giảm batch - báo lại để quyết định thủ công.\n{exc}")
        raise

    save_dir = Path(getattr(results, "save_dir", args.project) or args.project)

    split_json_path = Path(out_dir) / "split.json"
    if args.tile:
        # bản tiled dùng chung split.json gốc (chỉ tile thêm ảnh vào train)
        split_json_path = Path(out_dir) / "split.json"

    experiment = {
        "name": args.name,
        "model": args.model,
        "pretrained": args.pretrained,
        "imgsz": args.imgsz,
        "tile": args.tile,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "fraction": args.fraction,
        "all_data": args.all_data,
        "out_dir": out_dir,
        "train_hyperparams": TRAIN_HYPERPARAMS,
        "augmentations_repr": [repr(t) for t in train_kwargs["augmentations"]],
        "split_json": str(split_json_path),
        "split_json_sha256": _sha256_file(split_json_path),
        "transferred_weights_log": transferred_line,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    with open(save_dir / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(experiment, f, indent=2, ensure_ascii=False)
    print(f"\n📝 Đã ghi {save_dir / 'experiment.json'}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# 3 cấu hình cần chạy (dùng chung split.json trong yolo_dataset_v2/):
#
# A: python3 run_train.py --model yolo26n.pt \
#        --epochs 150 --patience 50 --batch 16 --name A_yolo26n_notile
#
# B: python3 run_train.py --model yolo26n.pt --tile \
#        --epochs 150 --patience 50 --batch 16 --name B_yolo26n_tile
#
# C: python3 run_train.py --model yolo26s-p2.yaml --pretrained yolo26s.pt \
#        --epochs 150 --patience 50 --batch 16 --name C_yolo26sp2
# ---------------------------------------------------------------------------
