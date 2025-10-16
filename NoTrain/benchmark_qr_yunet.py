# benchmark_qr_yunet.py
import argparse
import os
from time import perf_counter

import cv2
import numpy as np
from yunet_decode import YUNET  # 直接导入你已有的检测实现


def list_images(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
    return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(exts)])


def decode_qr_fullimg(img, detector):
    """整图用 OpenCV QR 解码（返回 bool，不影响计时逻辑）"""
    data, points, _ = detector.detectAndDecode(img)
    return bool(data)


def decode_qr_with_yunet(img,
                         yunet,
                         detector,
                         qr_class_id=3,
                         mode='640,640',
                         score_thresh=0.02,
                         save_dir=None,
                         basename=None,
                         allow_save_once=False):
    """
    先用 YUNET 检测，再在 ROI 上解码。
    - allow_save_once=True 时：如果 save_dir 与 basename 提供，则把第一个匹配 QR 类别的 ROI 保存一次。
    - 计时场景请将 allow_save_once=False（默认），以免保存影响计时。
    """
    bboxes, _ = yunet.detect(img, score_thresh=score_thresh, mode=mode)
    if len(bboxes) == 0:
        return False

    saved = False
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, score, cls_id = bbox
        if int(cls_id) != qr_class_id:
            continue
        xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
        roi = img[max(0, yi1):max(0, yi2), max(0, xi1):max(0, xi2)]
        if roi.size == 0:
            continue

        # 可选：保存 ROI（仅保存一次，且不在计时循环中调用）
        if (not saved) and allow_save_once and save_dir and basename:
            save_path = os.path.join(
                save_dir,
                f"{os.path.splitext(basename)[0]}_yunet_{i}.jpg"
            )
            try:
                cv2.imwrite(save_path, roi)
            except Exception:
                pass  # 保存失败不影响流程
            saved = True

        # 尝试在 ROI 内解码
        data, points, _ = detector.detectAndDecode(roi)
        if data:
            return True

    return False


def benchmark(image_dir,
              model_path,
              repeats=100,
              warmup=5,
              mode='640,640',
              score_thresh=0.02,
              qr_class_id=3):
    # 固定 OpenCV 线程数降低随机波动（需要可去掉）
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # 初始化
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    yunet = YUNET(model_path)
    qr_detector = cv2.QRCodeDetector()

    img_list = list_images(image_dir)
    if not img_list:
        print("No images found in folder:", image_dir)
        return

    print(f"Found {len(img_list)} images.")
    warm_img = cv2.imread(img_list[0])
    if warm_img is None:
        print(f"Failed to read warmup image: {img_list[0]}")
        return

    # --- 预热：两条管线各跑几次，剔除首次冷启动的影响 ---
    for _ in range(warmup):
        decode_qr_fullimg(warm_img, qr_detector)
        decode_qr_with_yunet(warm_img, yunet, qr_detector, qr_class_id, mode, score_thresh)

    # --- 正式计时 ---
    total_full, total_yunet = 0.0, 0.0
    n_valid = 0

    for idx, img_path in enumerate(img_list, 1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Skip] cannot read image: {img_path}")
            continue

        basename = os.path.basename(img_path)
        save_dir = os.path.dirname(img_path)

        # A: OpenCV 整图（计时）
        t0 = perf_counter()
        for _ in range(repeats):
            decode_qr_fullimg(img, qr_detector)
        t1 = perf_counter()
        avg_full = (t1 - t0) / repeats

        # B: YUNET 粗检测 → ROI 解码（计时）
        t0 = perf_counter()
        for _ in range(repeats):
            decode_qr_with_yunet(img, yunet, qr_detector, qr_class_id, mode, score_thresh)
        t1 = perf_counter()
        avg_yunet = (t1 - t0) / repeats

        # ✅ 在计时之外保存一次 ROI（若有），不影响计时
        decode_qr_with_yunet(img, yunet, qr_detector, qr_class_id, mode, score_thresh,
                             save_dir=save_dir, basename=basename, allow_save_once=True)

        total_full += avg_full
        total_yunet += avg_yunet
        n_valid += 1

        print(f"[{idx:03d}] {basename} | "
              f"OpenCV: {avg_full*1000:.3f} ms | "
              f"YUNET→OpenCV: {avg_yunet*1000:.3f} ms | "
              f"Speedup: {avg_full/avg_yunet:.2f}x")

    # --- 汇总 ---
    if n_valid == 0:
        print("No valid images processed.")
        return

    print("\n==== Summary ====")
    print(f"Images: {n_valid}, Repeats per image: {repeats}, Warmup: {warmup}")
    print(f"OpenCV avg:        {total_full/n_valid*1000:.3f} ms")
    print(f"YUNET→OpenCV avg:  {total_yunet/n_valid*1000:.3f} ms")
    print(f"Overall speedup:   {total_full/total_yunet:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: YUNET coarse detection vs. OpenCV QR decoding (with optional ROI saving)"
    )
    parser.add_argument("image_dir", help="Folder containing QR code images")
    parser.add_argument("--model", default="./yunet_n_640_640.onnx", help="Path to YUNET ONNX model file")
    parser.add_argument("--repeats", type=int, default=100, help="Repeat count per image for timing")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations before timing")
    parser.add_argument("--mode", default="640,640", help="YUNET input resize mode: 'ORIGIN'|'AUTO'|'W,H'")
    parser.add_argument("--score_thresh", type=float, default=0.02, help="Score threshold for YUNET filter")
    parser.add_argument("--qr_class_id", type=int, default=3, help="Class id for QR code in YUNET output")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"Image folder not found: {args.image_dir}")
        return

    benchmark(
        image_dir=args.image_dir,
        model_path=args.model,
        repeats=args.repeats,
        warmup=args.warmup,
        mode=args.mode,
        score_thresh=args.score_thresh,
        qr_class_id=args.qr_class_id
    )


if __name__ == "__main__":
    main()
