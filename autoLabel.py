"""
Hệ thống tự động gán nhãn ảnh dựa trên YOLO
Hỗ trợ 3 loại ảnh: ảnh băng rộng, ảng băng hẹp, ảnh tín hiệu WiFi
"""

import os
import time
import json
import ctypes
from pathlib import Path
import shutil
from typing import Dict, List, Tuple
import yaml

from dataset_split import build_dataset_once


def _configure_cuda_runtime():
    """Expose CUDA shared libraries installed by pip before torch loads."""
    site_packages = Path.home() / ".local" / "lib" / "python3.10" / "site-packages"
    cuda_lib_dir = site_packages / "nvidia" / "cu13" / "lib"
    nvrtc_builtins = cuda_lib_dir / "libnvrtc-builtins.so.13.0"

    if not nvrtc_builtins.exists():
        return

    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    cuda_lib_path = str(cuda_lib_dir)
    if cuda_lib_path not in current_ld_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{cuda_lib_path}:{current_ld_path}" if current_ld_path else cuda_lib_path
        )

    try:
        ctypes.CDLL(str(nvrtc_builtins), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass


_configure_cuda_runtime()

try:
    import numpy as np
    import cv2
except ImportError as exc:
    message = str(exc)
    if "numpy.core.multiarray failed to import" in message or "_ARRAY_API" in message:
        raise ImportError() from exc
    raise

from ultralytics import YOLO

class AutoLabelingSystem:
    def __init__(self, model_path: str = "yolo26n.pt", pretrained_weights: str = None,
                 conf_threshold: float = 0.5):
        """
        Khởi tạo hệ thống tự động gán nhãn

        Args:
            model_path: Đường dẫn mô hình YOLO (file .pt) hoặc file kiến trúc .yaml
                (vd "yolo26s-p2.yaml" - thêm đầu dò P2 để bắt tín hiệu nhỏ)
            pretrained_weights: Nếu model_path là .yaml, truyền tên/checkpoint .pt để
                load trọng số pretrained tương ứng (vd "yolo26s.pt")
            conf_threshold: Ngưỡng độ tin cậy tối thiểu
        """
        self.model = YOLO(model_path)
        if pretrained_weights:
            self.model.load(pretrained_weights)
        self.conf_threshold = conf_threshold
        self.detected_objects = {}

    def _find_images(self, images_dir: str) -> List[Path]:
        image_dir = Path(images_dir)
        image_extensions = {'.jpg', '.jpeg', '.png'}
        return sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        )
    
    def _load_class_names(self, classes_file: str) -> List[str]:
        path = Path(classes_file)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file classes: {classes_file}")
        with open(path, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
        if not names:
            raise ValueError(f"File classes rỗng: {classes_file}")
        return names

    def train_on_labeled_data(self,
                             labeled_images_dir: str = "data/rotate2",
                             labeled_annotations_dir: str = "data/labels_yolo",
                             classes_file: str = "data/classes.txt",
                             epochs: int = 50,
                             batch_size: int = 16,
                             imgsz: int = 640,
                             patience: int = 20,
                             project: str = "model_16-9",
                             name: str = "train",
                             tile_train_data: bool = False,
                             out_dir: str = "yolo_dataset_v2",
                             val_ratio: float = 0.15,
                             seed: int = 0,
                             test_dir: str = None,
                             **kwargs):
        """
        patience: số epoch liên tiếp không cải thiện mAP trên tập val trước khi
        dừng sớm (early stopping). Đặt patience=0 để tắt early stopping.
        tile_train_data: nếu True, cắt thêm tile 384x384 chồng lấn từ ảnh train
        (xem tile_dataset.py) để tăng cường phát hiện tín hiệu nhỏ. Áp dụng trên
        một bản SAO của out_dir (out_dir + "_tiled_<name>_<giờ>_<pid>") để không đụng tới out_dir
        gốc - val/test luôn giữ nguyên ảnh 640 đầy đủ.
        out_dir/val_ratio/seed/test_dir: truyền cho dataset_split.build_dataset_once
        để chia train/val (và test nếu có) MỘT LẦN CỐ ĐỊNH theo seed, dùng lại ở
        các lần train sau (xem out_dir/split.json) - tránh mỗi lần train ra một
        tập val khác nhau.
        """
        print("🔄 Chuẩn bị dữ liệu training...")

        split_info = build_dataset_once(
            images_dir=labeled_images_dir,
            labels_dir=labeled_annotations_dir,
            classes_file=classes_file,
            out_dir=out_dir,
            val_ratio=val_ratio,
            seed=seed,
            test_dir=test_dir,
        )
        data_yaml_path = Path(split_info["data_yaml"])

        # Tiling: cắt ảnh train thành các tile chồng lấn để tăng độ phân giải
        # tương đối cho tín hiệu nhỏ. Làm trên bản sao riêng (out_dir_tiled) -
        # không sửa out_dir gốc, val/test không bị tile.
        if tile_train_data:
            from tile_dataset import add_tiles_to_train
            # Mỗi lần chạy có thư mục tiled RIÊNG (name + thời gian + pid) và KHÔNG bao
            # giờ xoá thư mục cũ: xoá/dựng lại thư mục dùng chung sẽ làm crash run khác
            # đang đọc ảnh từ đó (FileNotFoundError trong DataLoader). copytree báo lỗi
            # nếu trùng tên thay vì ghi đè.
            tiled_dir = Path(f"{out_dir}_tiled_{name}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}")
            shutil.copytree(Path(out_dir), tiled_dir)
            n_tiles = add_tiles_to_train(tiled_dir)
            print(f"✅ Đã thêm {n_tiles} tile vào tập train (bản sao {tiled_dir}, không đụng {out_dir})")

            with open(data_yaml_path, "r", encoding="utf-8") as f:
                tiled_yaml = yaml.safe_load(f)
            tiled_yaml["path"] = str(tiled_dir.absolute())
            data_yaml_path = tiled_dir / "data.yaml"
            with open(data_yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(tiled_yaml, f)

        # Fine-tune model
        print("\n🚀 Bắt đầu fine-tuning YOLO...")
        results = self.model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=0,  # GPU device, thay đổi nếu cần
            patience=patience,  # early stopping
            save=True,
            verbose=True,
            project=project,
            name=name,
            **kwargs
        )

        save_dir = getattr(results, "save_dir", None)
        print("✅ Hoàn thành training!")
        if save_dir:
            print(f"📦 Mô hình đã lưu tại: {Path(save_dir) / 'weights' / 'best.pt'}")
        return results
    
    def predict_and_label(self, 
                         unlabeled_images_dir: str,
                         output_dir: str,
                         conf_threshold: float = None,
                         visualize_dir: str = None) -> Dict:
        """
        Dự đoán nhãn cho ảnh chưa được gán nhãn
        
        Args:
            unlabeled_images_dir: Thư mục chứa ảnh cần gán nhãn
            output_dir: Thư mục lưu kết quả
            conf_threshold: Ngưỡng độ tin cậy
            visualize_dir: Thư mục lưu ảnh có bounding box để kiểm tra trực quan
            
        Returns:
            Dictionary chứa kết quả dự đoán
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if visualize_dir:
            Path(visualize_dir).mkdir(parents=True, exist_ok=True)
        
        if not Path(unlabeled_images_dir).exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {unlabeled_images_dir}")

        image_files = self._find_images(unlabeled_images_dir)
        results_summary = {
            'total_images': len(image_files),
            'labeled_images': 0,
            'details': []
        }
        
        print(f"\n📸 Xử lý {len(image_files)} ảnh...")
        
        for idx, img_path in enumerate(image_files, 1):
            # Dự đoán vật thể
            results = self.model.predict(
                source=str(img_path),
                conf=conf_threshold,
                verbose=False
            )
            
            # Xử lý kết quả
            detections = []
            if results and len(results) > 0:
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            detection = {
                                'class': int(box.cls[0]),
                                'confidence': float(box.conf[0]),
                                'bbox': box.xyxy[0].tolist()
                            }
                            detections.append(detection)
            
            # Lưu nhãn (định dạng YOLO)
            if detections:
                label_path = Path(output_dir) / f"{img_path.stem}.txt"
                self._save_yolo_labels(str(img_path), detections, str(label_path))
                results_summary['labeled_images'] += 1

            if visualize_dir:
                output_viz_path = Path(visualize_dir) / img_path.name
                self._save_visualized_image(str(img_path), detections, str(output_viz_path))

            # Gộp các vệt P900 (frequency hopping) thành 1 sự kiện link nếu đủ >=3 vệt
            p900_summary = self.summarize_p900_activity(detections, str(img_path))

            results_summary['details'].append({
                'image': img_path.name,
                'detections': len(detections),
                'confidence_avg': np.mean([d['confidence'] for d in detections]) if detections else 0,
                'p900_link': p900_summary,
            })
            
            if idx % 10 == 0:
                print(f"  ✅ Đã xử lý {idx}/{len(image_files)} ảnh")
        
        print(f"\n✅ Gán nhãn hoàn thành!")
        self._print_summary(results_summary)
        
        return results_summary
    
    def _save_yolo_labels(self, image_path: str, detections: List[Dict], label_path: str):
        """
        Lưu nhãn theo định dạng YOLO
        """
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        with open(label_path, 'w') as f:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                
                # Chuyển đổi sang định dạng YOLO (center_x, center_y, width, height)
                center_x = (x1 + x2) / 2 / width
                center_y = (y1 + y2) / 2 / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                
                f.write(f"{det['class']} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n")

    def _get_class_name(self, class_id: int) -> str:
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _get_class_id_by_name(self, class_name: str):
        """Tra ngược tên lớp -> class_id. Không hardcode index vì thứ tự lớp
        trong classes.txt có thể đổi khi thêm/bớt lớp mới."""
        names = getattr(self.model, "names", {})
        items = names.items() if isinstance(names, dict) else enumerate(names)
        for cid, name in items:
            if name == class_name:
                return int(cid)
        return None

    def _load_freq_calibration(self, img_path: str):
        """Đọc calibration tần số (start/stop_frequency_hz) từ file <stem>.meta.json
        đi kèm ảnh (cùng thư mục ảnh, hoặc trong data/labels như với dữ liệu train).
        Trả về None nếu không tìm thấy - không suy đoán tần số từ tên file.
        """
        stem = Path(img_path).stem
        candidates = [
            Path(img_path).with_name(f"{stem}.meta.json"),
            Path("data/labels") / f"{stem}.meta.json",
        ]
        for meta_path in candidates:
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                start = meta.get("start_frequency_hz")
                stop = meta.get("stop_frequency_hz")
                width = meta.get("width")
                if start is not None and stop is not None and width:
                    return {"start_hz": start, "stop_hz": stop, "width": width}
        return None

    def _box_to_freq_mhz(self, x_center_px: float, calib: dict) -> float:
        """Quy đổi toạ độ X (trục tần số trong ảnh waterfall) sang MHz."""
        ratio = x_center_px / calib["width"]
        freq_hz = calib["start_hz"] + ratio * (calib["stop_hz"] - calib["start_hz"])
        return freq_hz / 1e6

    def summarize_p900_activity(self, detections: List[Dict], img_path: str, min_streaks: int = 3):
        """Gom các vệt (bounding box) P900 riêng lẻ thành một sự kiện "link P900
        đang hoạt động trên dải tần X-Y MHz" - P900 dùng frequency hopping nên
        1 phiên truyền thực tế thường tạo ra nhiều vệt rải trên nhiều tần số.

        Trả về dict tóm tắt (None nếu không đủ điều kiện) và tự in log.
        """
        p900_id = self._get_class_id_by_name("P900")
        if p900_id is None:
            return None

        p900_boxes = [d for d in detections if d["class"] == p900_id]
        if len(p900_boxes) < min_streaks:
            return None

        calib = self._load_freq_calibration(img_path)
        if calib is None:
            print(f"  📡 Phát hiện {len(p900_boxes)} vệt P900 trong {Path(img_path).name} "
                  f"(không có {Path(img_path).stem}.meta.json để quy đổi tần số)")
            return {"count": len(p900_boxes), "freq_min_mhz": None, "freq_max_mhz": None}

        freqs = [self._box_to_freq_mhz((b["bbox"][0] + b["bbox"][2]) / 2, calib) for b in p900_boxes]
        freq_min, freq_max = min(freqs), max(freqs)
        print(f"  📡 Phát hiện P900 hoạt động ({len(p900_boxes)} vệt), "
              f"dải {freq_min:.1f}–{freq_max:.1f} MHz [{Path(img_path).name}]")
        return {"count": len(p900_boxes), "freq_min_mhz": freq_min, "freq_max_mhz": freq_max}

    def _save_visualized_image(self, image_path: str, detections: List[Dict], output_path: str):
        """
        Lưu ảnh có bounding box, tên nhãn và confidence để kiểm tra trực quan.
        """
        img = cv2.imread(image_path)
        if img is None:
            return

        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]

        for det in detections:
            class_id = det['class']
            confidence = det['confidence']
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            color = colors[class_id % len(colors)]
            label = f"{self._get_class_name(class_id)} {confidence:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_w, text_h = text_size
            y_text = max(y1, text_h + baseline + 4)
            cv2.rectangle(
                img,
                (x1, y_text - text_h - baseline - 4),
                (x1 + text_w + 6, y_text),
                color,
                -1
            )
            cv2.putText(
                img,
                label,
                (x1 + 3, y_text - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.imwrite(output_path, img)
    
    def visualize_predictions(self, 
                             images_dir: str,
                             labels_dir: str,
                             output_viz_dir: str,
                             num_samples: int = 10):
        """
        Trực quan hóa kết quả dự đoán
        """
        Path(output_viz_dir).mkdir(parents=True, exist_ok=True)
        
        image_files = self._find_images(images_dir)[:num_samples]
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        
        print(f"\n🎨 Trực quan hóa {len(image_files)} ảnh...")
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            label_path = Path(labels_dir) / f"{img_path.stem}.txt"
            
            if label_path.exists():
                height, width = img.shape[:2]
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            center_x = float(parts[1]) * width
                            center_y = float(parts[2]) * height
                            box_width = float(parts[3]) * width
                            box_height = float(parts[4]) * height
                            
                            x1 = int(center_x - box_width / 2)
                            y1 = int(center_y - box_height / 2)
                            x2 = int(center_x + box_width / 2)
                            y2 = int(center_y + box_height / 2)
                            
                            color = colors[class_id % len(colors)]
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img, self._get_class_name(class_id), (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            output_path = Path(output_viz_dir) / img_path.name
            cv2.imwrite(str(output_path), img)
        
        print(f"✅ Đã lưu hình ảnh trực quan hóa tới: {output_viz_dir}")
    
    def _print_summary(self, summary: Dict):
        """In tóm tắt kết quả"""
        print("\n" + "="*50)
        print("📊 TÓM TẮT KẾT QUẢ")
        print("="*50)
        print(f"Tổng số ảnh: {summary['total_images']}")
        print(f"Ảnh được gán nhãn: {summary['labeled_images']}")
        if summary['total_images'] == 0:
            print("Tỷ lệ: 0.0%")
            print("Không tìm thấy ảnh đầu vào để xử lý.")
        else:
            print(f"Tỷ lệ: {summary['labeled_images']/summary['total_images']*100:.1f}%")
        print("="*50 + "\n")


def main():
    # yolov8m.pt: mô hình YOLOv8 size "m" (medium), pretrained trên COCO
    system = AutoLabelingSystem( model_path="yolo26s-p2.yaml",
        pretrained_weights="yolo26s.pt",
        conf_threshold=0.5,)

    # 1. Fine-tune trên dữ liệu mới (data/img + data/labels_yolo, theo data/classes.txt)
    print("=" * 60)
    print("BƯỚC 1: FINE-TUNE YOLO26 TRÊN DỮ LIỆU MỚI")
    print("=" * 60)

    from augmentations import build_custom_augmentations

    system.train_on_labeled_data(
        labeled_images_dir="data/img",
        labeled_annotations_dir="data/labels_yolo",
        classes_file="data/classes.txt",
        epochs=150,
        batch_size=8,
        tile_train_data=True,  # cắt tile 384x384 chồng lấn cho tập train (tile_dataset.py)
        # Màu: không đổi hue/saturation vì colormap mã hóa công suất
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,        # mô phỏng thay đổi gain / reference level
        # Hình học: không xoay, không shear, vì 2 trục (tần số, thời gian) có ý nghĩa vật lý
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        translate=0.2,    # dịch x = đổi center freq, dịch y = lệch thời điểm, đều hợp lệ
        scale=0.1,        # nhỏ, để giữ băng thông và độ dài burst
        flipud=0.5,       # đảo thời gian: an toàn
        fliplr=0.0,       # đảo phổ: không có tương đương vật lý -> tắt
        mosaic=0.0,
        mixup=0.0,
        # Nhiễu nền + dịch tần số + các loại blur mô phỏng RBW/FFT window/nén frame_width
        # (xem augmentations.py) qua Albumentations
        augmentations=build_custom_augmentations(),
    )
    
    # 2. Tự động gán nhãn cho ảnh mới
    # print("\n" + "=" * 60)
    # print("BƯỚC 2: TỰ ĐỘNG GÁN NHÃN CHO ẢNH MỚI")
    # print("=" * 60)
    
    # results = system.predict_and_label(
    #     unlabeled_images_dir="data_22_8_26/images",
    #     output_dir="output_labels",
    #     conf_threshold=0.5,
    #     visualize_dir="visualized_results"
    # )
    
    # # 3. Trực quan hóa kết quả
    # print("\n" + "=" * 60)
    # print("BƯỚC 3: TRỰC QUAN HÓA KẾT QUẢ")
    # print("=" * 60)
    
    # system.visualize_predictions(
    #     images_dir="data/rotate2",
    #     labels_dir="output_labels",
    #     output_viz_dir="visualized_results",
    #     num_samples=20
    # )
    
    # # Lưu kết quả
    # with open("labeling_results.json", 'w') as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)
    
    # print("\n✅ Toàn bộ quá trình hoàn thành!")


if __name__ == "__main__":
    main()
