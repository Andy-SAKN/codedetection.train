# -*- coding: utf-8 -*-
import os
import glob
import yaml
import cv2
import numpy as np

def load_sr_engine(root_dir=".", yaml_name="qbar_sr.yaml"):
    """
    读取 qbar_sr.yaml 并加载 onnx 为 OpenCV DNN 网络。
    要求 yaml 至少包含键 onnx_path。
    """
    yml = os.path.join(root_dir, yaml_name)
    if not os.path.exists(yml):
        raise FileNotFoundError(f"找不到配置文件: {yml}")

    with open(yml, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    onnx_path = cfg.get("onnx_path", None)
    if not onnx_path:
        raise KeyError("qbar_sr.yaml 缺少 onnx_path 字段")
    onnx_path = os.path.join(root_dir, onnx_path)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"找不到 ONNX 模型: {onnx_path}")

    net = cv2.dnn.readNetFromONNX(onnx_path)
    return net

def run_sr_on_gray(net, gray_u8):
    """
    对单通道灰度图做一次 SR 推理。
    约定：输入 gray_u8 为 HxW uint8。
    预处理：与您现有示例一致（不做归一化/减均值），仅转换为 float32。
    如果你的 SR 训练做过归一化，请在此处对齐。
    """
    assert gray_u8.ndim == 2 and gray_u8.dtype == np.uint8
    blob = cv2.dnn.blobFromImage(gray_u8.astype(np.float32))  # shape: (1,1,H,W)
    net.setInput(blob)
    out = net.forward()                # 可能是 (1,1,H',W') / (1,H',W') / (H',W')
    out = np.squeeze(out)              # 压到 2D
    if out.ndim == 3:                  # 若仍是 (C,H,W) 就取第0通道
        out = out[0]
    if out.ndim != 2:
        raise RuntimeError(f"SR 输出维度异常: {out.shape}")
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def decode_qr(gray_u8):
    """
    使用 OpenCV 的 QRCodeDetector 解码。
    先尝试多码接口，失败回退单码。
    返回列表 [(text, points)], points 为 4x2 顶点；失败返回 []。
    """
    det = cv2.QRCodeDetector()
    # 尝试多码（4.7+）
    try:
        ok, infos, points, _ = det.detectAndDecodeMulti(gray_u8)
        if ok and points is not None:
            out = []
            for txt, pts in zip(infos, points):
                out.append(((txt or "").strip(), pts))
            if out:
                return out
    except Exception:
        pass
    # 回退单码
    txt, pts, _ = det.detectAndDecode(gray_u8)
    if txt:
        return [(txt.strip(), pts)]
    return []

def draw_results(bgr, results, color=(0, 255, 0)):
    """
    在图上绘制二维码四边形与解码文本。
    """
    vis = bgr.copy()
    for txt, pts in results:
        if pts is not None and len(pts) == 4:
            pts_i = np.int32(pts).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts_i], True, color, 2, cv2.LINE_AA)
            p = tuple(np.int32(pts[0]))
            p = (max(0, p[0]), max(18, p[1]))
            cv2.putText(vis, txt[:80], p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis

def process_image(img_path, sr_net):
    """
    对单张图片进行：
      1) 原图灰度解码
      2) 若失败，则整图灰度 SR 后再解码
    返回 (stage, results, vis_bgr)，stage 为 "raw" 或 "sr" 或 "fail"
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise RuntimeError(f"无法读取图片: {img_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 先原图解
    res_raw = decode_qr(gray)
    if res_raw:
        return "raw", res_raw, draw_results(bgr, res_raw, (0, 200, 0))

    # 原图失败 -> SR
    sr_gray = run_sr_on_gray(sr_net, gray)
    res_sr = decode_qr(sr_gray)
    if res_sr:
        # 可视化：在 SR 灰度图上画框更直观（因为 SR 改变了尺寸）
        sr_bgr = cv2.cvtColor(sr_gray, cv2.COLOR_GRAY2BGR)
        return "sr", res_sr, draw_results(sr_bgr, res_sr, (255, 0, 0))

    return "fail", [], bgr

def main():
    sr_net = load_sr_engine(".", "qbar_sr.yaml")
    os.makedirs("sr_decode_vis", exist_ok=True)

    img_paths = []
    img_paths += glob.glob("images/*.jpg")
    img_paths += glob.glob("images/*.JPG")
    img_paths += glob.glob("images/*.jpeg")
    img_paths += glob.glob("images/*.png")
    if not img_paths:
        print("images 目录为空。")
        return

    for p in img_paths:
        stage, results, vis = process_image(p, sr_net)
        name = os.path.splitext(os.path.basename(p))[0]
        save_path = os.path.join("sr_decode_vis", f"{name}_{stage}.png")
        cv2.imwrite(save_path, vis)

        if stage == "fail":
            print(f"[{name}] 原图与 SR 后均未解出。可视化: {save_path}")
        else:
            print(f"[{name}] {('原图' if stage=='raw' else 'SR 后')}解码成功 {len(results)} 个，保存: {save_path}")
            for i, (txt, _) in enumerate(results, 1):
                print(f"  -> 结果{i}: {txt}")

if __name__ == "__main__":
    main()
