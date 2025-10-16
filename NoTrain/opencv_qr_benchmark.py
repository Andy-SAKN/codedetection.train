# opencv_qr_benchmark.py
import argparse
import os
from time import perf_counter
from glob import glob

import cv2
import numpy as np


def list_images(path):
    """支持传入单图或目录。返回图像路径列表。"""
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
    if os.path.isdir(path):
        files = []
        for ext in exts:
            files.extend(glob(os.path.join(path, f"*{ext}")))
        return sorted(files)
    elif os.path.isfile(path) and path.lower().endswith(exts):
        return [path]
    return []


def decode_qr_fullimg(img, detector):
    """
    直接整图用 OpenCV QR 解码。
    返回: (ok, data(str), points(np.ndarray|None))
    """
    data, points, _ = detector.detectAndDecode(img)
    return bool(data), (data or ""), points


def draw_qr(img, points, label_text="QR"):
    """在图上画多边形框 & 标签（若 points 存在）"""
    if points is None:
        return img
    pts = points.reshape(-1, 2).astype(int)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    # 放标签在第一个点附近
    x, y = pts[0]
    cv2.putText(img, label_text, (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def save_annotated(img_path, ok, data, points, out_suffix="_opencv"):
    """
    在不影响计时的情况下保存可视化图：
    - 只保存一次（每张原图一次），不放进计时循环
    - 命名: 原名 + out_suffix + 扩展名
    """
    dirname = os.path.dirname(img_path)
    basename = os.path.basename(img_path)
    name, ext = os.path.splitext(basename)
    out_path = os.path.join(dirname, f"{name}{out_suffix}{ext}")

    img = cv2.imread(img_path)
    if img is None:
        return
    label = "QR:OK" if ok else "QR:FAIL"
    img = draw_qr(img, points, label_text=label)
    # 如果成功且有解码内容，顺带打印在图上（避免过长截断）
    if ok and data:
        txt = (data[:48] + "...") if len(data) > 48 else data
        cv2.putText(img, txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    try:
        cv2.imwrite(out_path, img)
    except Exception:
        pass


def benchmark(image_path_or_dir, repeats=100, warmup=5, save_vis=True):
    # 降低随机波动（如有需要可注释掉）
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    img_list = list_images(image_path_or_dir)
    if not img_list:
        print(f"No valid images found in: {image_path_or_dir}")
        return

    print(f"Found {len(img_list)} images.")
    qr_detector = cv2.QRCodeDetector()

    # 预热
    warm_img = cv2.imread(img_list[0])
    if warm_img is None:
        print(f"[Skip warmup] Failed to read: {img_list[0]}")
    else:
        for _ in range(warmup):
            decode_qr_fullimg(warm_img, qr_detector)

    total_time = 0.0
    n_valid = 0

    header = f"{'Image':<30} {'Class':<8} {'Success':<7} {'Avg(ms)':>9}  {'SuccessRate':>12}  {'ExampleData'}"
    print(header)
    print("-" * len(header))

    for img_path in img_list:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Skip] cannot read image: {img_path}")
            continue

        # 计时循环
        ok_count = 0
        t0 = perf_counter()
        last_ok, last_data, last_points = False, "", None
        for _ in range(repeats):
            ok, data, points = decode_qr_fullimg(img, qr_detector)
            if ok:
                ok_count += 1
                last_ok, last_data, last_points = ok, data, points
        t1 = perf_counter()

        avg_t_ms = (t1 - t0) / repeats * 1000.0
        total_time += (t1 - t0)
        n_valid += 1

        # 计时外保存一份可视化
        if save_vis:
            save_annotated(img_path, last_ok, last_data, last_points, out_suffix="_opencv")

        # 输出一行结果（类别恒为 QR）
        success_rate = f"{ok_count}/{repeats}"
        example = (last_data[:60] + "...") if last_data and len(last_data) > 60 else (last_data or "-")
        print(f"{os.path.basename(img_path):<30} {'QR':<8} {('Y' if ok_count>0 else 'N'):<7} "
              f"{avg_t_ms:>9.3f}  {success_rate:>12}  {example}")

    # 汇总
    if n_valid == 0:
        print("No valid images processed.")
        return

    print("\n==== Summary ====")
    print(f"Images: {n_valid}, Repeats per image: {repeats}, Warmup: {warmup}")
    print(f"OpenCV (QR) overall average: { (total_time / n_valid) * 1000.0 / repeats :.3f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Pure OpenCV QR decoding benchmark (no YUNET): outputs class, decode status, timing, etc."
    )
    parser.add_argument("path", help="Image file or folder containing QR code images")
    parser.add_argument("--repeats", type=int, default=100, help="Repeat count per image for timing")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations before timing")
    parser.add_argument("--no-save", action="store_true", help="Do not save annotated images")
    args = parser.parse_args()

    benchmark(args.path, repeats=args.repeats, warmup=args.warmup, save_vis=(not args.no_save))


if __name__ == "__main__":
    main()
