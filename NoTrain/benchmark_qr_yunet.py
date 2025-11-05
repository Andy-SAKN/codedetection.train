# benchmark_qr_yunet.py
import argparse
import os
from time import perf_counter

import cv2
import numpy as np
from yunet_decode import YUNET  # 直接导入你已有的检测实现

# 不在模块导入时加载 SR；仅在 main 且用户启用 --use_sr 时动态导入/加载
load_sr_engine = None
run_sr_on_gray = None


def list_images(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.gif')
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
                         allow_save_once=False,
                         sr_net=None,
                         expand_ratio=1.2):  # 新增参数 expand_ratio
    """
    先用 YUNET 检测，再在 ROI 上解码。
    返回 (success: bool, used_sr: bool, sr_attempted: bool)
    - expand_ratio: 在原始 bbox 基础上的扩大比例，例如 1.2 表示 120%
    - sr_attempted: 表示是否在本次处理中曾尝试调用 SR（即解码失败后实际调用过 SR 推理）
    """
    bboxes, _ = yunet.detect(img, score_thresh=score_thresh, mode=mode)
    if len(bboxes) == 0:
        return False, False, False

    saved = False
    sr_any_attempted = False

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, score, cls_id = bbox
        if int(cls_id) != qr_class_id:
            continue

        # 按原图坐标进行扩大（参考 yunet_decode.py 中的做法）
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        w = x2i - x1i
        h = y2i - y1i
        if w <= 0 or h <= 0:
            continue
        cx = x1i + w // 2
        cy = y1i + h // 2
        new_w = int(w * expand_ratio)
        new_h = int(h * expand_ratio)
        new_x1 = max(0, cx - new_w // 2)
        new_y1 = max(0, cy - new_h // 2)
        new_x2 = min(img.shape[1], cx + new_w // 2)
        new_y2 = min(img.shape[0], cy + new_h // 2)

        xi1, yi1, xi2, yi2 = new_x1, new_y1, new_x2, new_y2

        roi = img[yi1:yi2, xi1:xi2]
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
                # 仅在保存/打印路径输出扩大的比例和最终坐标，避免影响计时
                print(f"[YUNET] bbox expanded by {expand_ratio:.2f}x -> ({xi1},{yi1},{xi2},{yi2}), saved: {save_path}")
            except Exception:
                pass  # 保存失败不影响流程
            saved = True

        # 尝试在 ROI 内解码（直接用 OpenCV）
        data, points, _ = detector.detectAndDecode(roi)
        if data:
            return True, False, sr_any_attempted

        # ROI 解码失败 -> 若提供 SR 模型则对 ROI 做超分再尝试一次
        if sr_net is not None and run_sr_on_gray is not None:
            sr_any_attempted = True
            try:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                sr_gray = run_sr_on_gray(sr_net, gray_roi)
                # sr_gray 为 uint8 灰度图，直接用 detector 解码
                data2, points2, _ = detector.detectAndDecode(sr_gray)
                if data2:
                    return True, True, True
            except Exception:
                # SR 失败不影响流程，继续下一个 bbox
                pass

    return False, False, sr_any_attempted


def benchmark(image_dir,
              model_path,
              repeats=3,
              warmup=5,
              mode='640,640',
              score_thresh=0.02,
              qr_class_id=3,
              sr_net=None,
              save_after=False,
              expand_ratio=1.2):  # 新增参数 expand_ratio
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

    # 仅当用户请求保存时才创建 after_images（与 images 并列）
    after_dir = None
    if save_after:
        after_dir = os.path.join(os.path.dirname(os.path.abspath(image_dir)), "after_images")
        try:
            os.makedirs(after_dir, exist_ok=True)
        except Exception:
            pass

    print(f"Found {len(img_list)} images.")
    warm_img = cv2.imread(img_list[0])
    if warm_img is None:
        print(f"Failed to read warmup image: {img_list[0]}")
        return

    # --- 预热：两条管线各跑几次，剔除首次冷启动的影响 ---
    for _ in range(warmup):
        decode_qr_fullimg(warm_img, qr_detector)
        decode_qr_with_yunet(warm_img, yunet, qr_detector, qr_class_id, mode, score_thresh, sr_net=sr_net, expand_ratio=expand_ratio)

    # --- 正式计时 ---
    total_full, total_yunet = 0.0, 0.0
    n_valid = 0

    # 新增统计：成功计数
    full_success = 0
    yunet_success = 0
    yunet_success_with_sr = 0
    yunet_sr_attempts = 0  # 新增：统计 SR 实际尝试次数

    for idx, img_path in enumerate(img_list, 1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Skip] cannot read image: {img_path}")
            continue

        basename = os.path.basename(img_path)

        # A: OpenCV 整图（计时）
        t0 = perf_counter()
        for _ in range(repeats):
            decode_qr_fullimg(img, qr_detector)
        t1 = perf_counter()
        avg_full = (t1 - t0) / repeats

        # B: YUNET 粗检测 → ROI 解码（计时）
        t0 = perf_counter()
        for _ in range(repeats):
            # 在计时中也传入 sr_net 与 expand_ratio（但不进行保存/打印）
            decode_qr_with_yunet(img, yunet, qr_detector, qr_class_id, mode, score_thresh, sr_net=sr_net, expand_ratio=expand_ratio)
        t1 = perf_counter()
        avg_yunet = (t1 - t0) / repeats

        # ✅ 在计时之外保存一次 ROI（若用户请求且有 ROI），并统计成功率
        full_ok = decode_qr_fullimg(img, qr_detector)
        if after_dir is not None:
            yunet_ok, yunet_used_sr, sr_attempted = decode_qr_with_yunet(
                img, yunet, qr_detector, qr_class_id, mode, score_thresh,
                save_dir=after_dir, basename=basename, allow_save_once=True, sr_net=sr_net, expand_ratio=expand_ratio
            )
        else:
            yunet_ok, yunet_used_sr, sr_attempted = decode_qr_with_yunet(
                img, yunet, qr_detector, qr_class_id, mode, score_thresh,
                save_dir=None, basename=None, allow_save_once=False, sr_net=sr_net, expand_ratio=expand_ratio
            )

        # 统计
        if full_ok:
            full_success += 1
        if yunet_ok:
            yunet_success += 1
            if yunet_used_sr:
                yunet_success_with_sr += 1
        if sr_attempted:  # 仅在尝试过 SR 时计数
            yunet_sr_attempts += 1

        total_full += avg_full
        total_yunet += avg_yunet
        n_valid += 1

        # 打印摘要（此打印发生在计时之外）
        print(f"[{idx:03d}] {basename} | "
              f"OpenCV: {avg_full*1000:.3f} ms | "
              f"YUNET→OpenCV: {avg_yunet*1000:.3f} ms | "
              f"Speedup: {avg_full/avg_yunet:.2f}x | "
              f"OpenCV_OK: {str(full_ok)} | YUNET_OK: {str(yunet_ok)}{'(SR)' if yunet_used_sr else ''}")

    # --- 汇总 ---
    if n_valid == 0:
        print("No valid images processed.")
        return

    print("\n==== Summary ====")
    print(f"Images processed: {n_valid}, Repeats per image: {repeats}, Warmup: {warmup}")
    print(f"OpenCV avg:        {total_full/n_valid*1000:.3f} ms")
    print(f"YUNET→OpenCV avg:  {total_yunet/n_valid*1000:.3f} ms")
    print(f"Overall speedup:   {total_full/total_yunet:.2f}x")
    # 成功率
    print("\n==== Success Rate ====")
    print(f"OpenCV (full image): {full_success}/{n_valid} ({full_success/n_valid*100:.2f}%)")
    print(f"YUNET pipeline:      {yunet_success}/{n_valid} ({yunet_success/n_valid*100:.2f}%)")
    # 仅当用户启用了 SR（即传入了 --use_sr 并且主函数尝试加载）时 sr_net 会非 None，才显示 SR 相关统计
    if sr_net is not None:
        denom = yunet_sr_attempts
        pct = (yunet_success_with_sr / denom * 100) if denom > 0 else 0.0
        print(f"YUNET successes that used SR: {yunet_success_with_sr}/{denom} ({pct:.2f}%)")

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: YUNET coarse detection vs. OpenCV QR decoding (with optional ROI saving)"
    )
    parser.add_argument("image_dir", help="Folder containing QR code images")
    parser.add_argument("--model", default="./yunet_n_640_640.onnx", help="Path to YUNET ONNX model file")
    parser.add_argument("--repeats", type=int, default=3, help="Repeat count per image for timing")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations before timing")
    parser.add_argument("--mode", default="640,640", help="YUNET input resize mode: 'ORIGIN'|'AUTO'|'W,H'")
    parser.add_argument("--score_thresh", type=float, default=0.02, help="Score threshold for YUNET filter")
    parser.add_argument("--qr_class_id", type=int, default=3, help="Class id for QR code in YUNET output")
    # 新增 SR 相关参数
    parser.add_argument("--use_sr", action="store_true", help="Enable SR fallback using qbar_sr.yaml / sr.py")
    parser.add_argument("--sr_root", default=".", help="Root dir for SR yaml / onnx paths")
    parser.add_argument("--sr_yaml", default="qbar_sr.yaml", help="SR yaml filename (relative to sr_root)")
    # 新增：是否保存 after_images
    parser.add_argument("--save_after", action="store_true", help="Save one ROI per image to after_images directory (placed alongside images folder)")
    parser.add_argument("--expand_ratio", type=float, default=1.2, help="BBox expand ratio (e.g. 1.2 for 120%)")  # 新增参数
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"Image folder not found: {args.image_dir}")
        return

    # 仅在 --use_sr 时动态导入并加载 SR（否则不导入、不打印）
    sr_net = None
    if args.use_sr:
        try:
            import sr as sr_module  # 动态导入
            global load_sr_engine, run_sr_on_gray
            load_sr_engine = getattr(sr_module, "load_sr_engine", None)
            run_sr_on_gray = getattr(sr_module, "run_sr_on_gray", None)
        except Exception:
            print("SR module not available (could not import sr.py). Continuing without SR.")
            load_sr_engine = None
            run_sr_on_gray = None

        if load_sr_engine is not None:
            try:
                sr_net = load_sr_engine(args.sr_root, args.sr_yaml)
                print("SR engine loaded and enabled for fallback.")
            except Exception as e:
                print(f"Failed to load SR engine: {e}. Continuing without SR.")
                sr_net = None

    benchmark(
        image_dir=args.image_dir,
        model_path=args.model,
        repeats=args.repeats,
        warmup=args.warmup,
        mode=args.mode,
        score_thresh=args.score_thresh,
        qr_class_id=args.qr_class_id,
        sr_net=sr_net,
        save_after=args.save_after,
        expand_ratio=args.expand_ratio  # 传递 expand_ratio
    )

if __name__ == "__main__":
    main()
