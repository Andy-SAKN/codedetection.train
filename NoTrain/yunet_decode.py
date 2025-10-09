import argparse
import os
from time import time

import cv2
import numpy as np
import onnx
import onnxruntime




def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def sigmoid(x):
    sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
    return sig


def nms(dets, thresh, opencv_mode=True):
    if opencv_mode:
        _boxes = dets[:, :4].copy()
        scores = dets[:, -1]
        _boxes[:, 2] = _boxes[:, 2] - _boxes[:, 0]
        _boxes[:, 3] = _boxes[:, 3] - _boxes[:, 1]
        keep = cv2.dnn.NMSBoxes(
            bboxes=_boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=0.,
            nms_threshold=thresh,
            eta=1,
            top_k=5000)
        if len(keep) > 0:
            return keep.flatten()
        else:
            return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, -1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clip(min=0, max=max_shape[1])
        y1 = y1.clip(min=0, max=max_shape[0])
        x2 = x2.clip(min=0, max=max_shape[1])
        y2 = y2.clip(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[..., i % 2] + distance[..., i]
        py = points[..., i % 2 + 1] + distance[..., i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def resize_img(img, mode):
    if mode == 'ORIGIN':
        det_img, det_scale = img, 1.
    elif mode == 'AUTO':
        assign_h = ((img.shape[0] - 1) & (-32)) + 32
        assign_w = ((img.shape[1] - 1) & (-32)) + 32
        det_img = np.zeros((assign_h, assign_w, 3), dtype=np.uint8)
        det_img[:img.shape[0], :img.shape[1], :] = img
        det_scale = 1.
    else:
        if mode == 'VGA':
            input_size = (640, 480)
        else:
            input_size = list(map(int, mode.split(',')))
        assert len(input_size) == 2
        x, y = max(input_size), min(input_size)
        if img.shape[1] > img.shape[0]:
            input_size = (x, y)
        else:
            input_size = (y, x)
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

    return det_img, det_scale


def draw(img, bboxes, kpss, out_path, with_kps=True):
    # # 检查是否有条形码类别
    # has_barcode = np.any(bboxes[:, 5] == 1) if bboxes.shape[0] > 0 else False
    # if has_barcode:
    #     # 直接用整图解码并绘制
    #     def decode_barcode(image):
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         barcodes = pyzbar.decode(gray)
    #         for barcode in barcodes:
    #             (x, y, w, h) = barcode.rect
    #             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #             barcode_data = barcode.data.decode("utf-8")
    #             barcode_type = barcode.type
    #             text = "{} ({})".format(barcode_data, barcode_type)
    #             cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #             print("[INFO] Found barcode: {}, {}".format(barcode_data, barcode_type))
    #         cv2.imshow("Barcode Reader", image)
    #         cv2.waitKey(0)
    #     decode_barcode(img)
    # 其它类别正常绘制和解码
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        score = bbox[4]
        cls_id = int(bbox[5])
        if cls_id != 1:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f'cls:{cls_id} conf:{score:.2f}'
            cv2.putText(
                img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            # 二维码解码逻辑：如果cls_id==3，尝试解码并输出
            if cls_id == 3:
                qr_roi = img[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
                if qr_roi.size > 0:
                    qr_decoder = cv2.QRCodeDetector()
                    data, points, _ = qr_decoder.detectAndDecode(qr_roi)
                    if data:
                        print(f"[QR] 检测到二维码内容: {data}")
                    else:
                        print("[QR] 检测到cls=3但未能解码二维码")
            # DataMatrix解码逻辑：如果cls_id==5，尝试解码并输出
            if cls_id == 5:
                dm_roi = img[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
                if dm_roi.size > 0:
                    gray_dm = cv2.cvtColor(dm_roi, cv2.COLOR_BGR2GRAY)
                    results = pylibdmtx.decode(gray_dm)
                    if results:
                        for r in results:
                            try:
                                dm_data = r.data.decode("utf-8")
                            except Exception:
                                dm_data = str(r.data)
                            print(f"[DataMatrix] 检测到内容: {dm_data}")
                    else:
                        print("[DataMatrix] 检测到cls=5但未能解码DataMatrix")
    print('Detection result saved to:', out_path)
    cv2.imwrite(out_path, img)




class Timer:

    def __init__(self) -> None:
        self.total = 0
        self.val = 0
        self.epochs = 0
        self.istic = False
        self.mode = 's'

    def tic(self):
        assert not self.istic
        self.istic = True
        self.val = time()

    def toc(self):
        assert self.istic
        self.istic = False
        self.epochs += 1
        self.total += time() - self.val
        self.val = 0

    def total_second(self):
        return self.total

    def average_second(self):
        return self.total / self.epochs

    def reset(self):
        self.total = 0
        self.val = 0
        self.epochs = 0
        self.istic = False

    def set_mode(self, mode='s'):
        assert mode in ('s', 'ms')
        if mode == 's' and self.mode == 'ms':
            self.total /= 1000.
        elif mode == 'ms' and self.mode == 's':
            self.total *= 1000.


class TimeEngine:

    def __init__(self):
        self.container = {}

    def tic(self, key):
        if self.container.get(key, None) is None:
            self.container[key] = Timer()
        self.container[key].tic()

    def toc(self, key):
        assert key in self.container
        self.container[key].toc()

    def total_second(self, key=None):
        if key is None:
            total_s = 0
            for k, v in self.container.items():
                total_s += v.total_second()
            return total_s
        else:
            return self.container[key].total_second()

    def average_second(self, key):
        return self.container[key].average_second()

    def reset(self, key=None):
        if key:
            self.container[key].reset()
        else:
            self.container = {}

    def set_mode(self, mode='s'):
        for k, v in self.container.items():
            v.set_mode(mode)


class Detector:

    def __init__(self, model_file=None, nms_thresh=0.5) -> None:
        self.model_file = model_file
        self.nms_thresh = nms_thresh
        assert os.path.exists(self.model_file)
        model = onnx.load(model_file)
        onnx.checker.check_model(model)
        self.session = onnxruntime.InferenceSession(self.model_file, None)
        self.time_engine = TimeEngine()

    def preprocess(self, img):
        pass

    def forward(self, img, score_thresh):
        pass

    def detect(self, img, score_thresh=0.5, mode='ORIGIN'):
        pass


class YUNET(Detector):

    def __init__(self, model_file=None, nms_thresh=0.5) -> None:
        super().__init__(model_file, nms_thresh)
        self.taskname = 'yunet'
        self.priors_cache = []
        self.strides = [8, 16, 32]
        self.NK = 4 # original 5



    def forward(self, img, score_thresh):
        input_size = tuple(img.shape[0:2][::-1])  # (W, H)

        blob = np.transpose(img, [2, 0, 1]).astype(np.float32)[np.newaxis, ...].copy()
        # print(f"[DEBUG] Input blob shape: {blob.shape}")

        # inference
        nets_out = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        # for i, out in enumerate(nets_out):
            # print(f"[DEBUG] nets_out[{i}]: shape = {out.shape}")

        scores, bboxes, kpss, labels = [], [], [], []
        for idx, stride in enumerate(self.strides):
            # print(f"\n[INFO] === Processing stride {stride} (idx={idx}) ===")

            # classification prediction (1, N, C) → squeeze → (N, C)
            cls_pred_raw = nets_out[idx].squeeze(0)
            obj_pred = nets_out[idx + len(self.strides)].reshape(-1, 1)
            reg_pred = nets_out[idx + len(self.strides) * 2].reshape(-1, 4)
            kps_pred = nets_out[idx + len(self.strides) * 3].reshape(-1, self.NK * 2)

            # check shape consistency
            if cls_pred_raw.shape[0] != obj_pred.shape[0]:
                print(f"[WARNING] Shape mismatch at idx={idx}, skipping this scale.")
                continue

            # classification score: max class score for each anchor
            cls_score = np.max(cls_pred_raw, axis=1, keepdims=True)  # shape: (N, 1)
            cls_label = np.argmax(cls_pred_raw, axis=1)  # shape: (N,)

            # construct anchor center
            anchor_centers = np.stack(
                np.mgrid[:(input_size[1] // stride), :(input_size[0] // stride)][::-1],
                axis=-1
            )
            anchor_centers = (anchor_centers * stride).astype(np.float32).reshape(-1, 2)
            # print(f"[DEBUG] stride={stride} | kps_pred.shape={kps_pred.shape} | anchors={anchor_centers.shape}")

            bbox_cxy = reg_pred[:, :2] * stride + anchor_centers
            bbox_wh = np.exp(reg_pred[:, 2:]) * stride
            tl_x = bbox_cxy[:, 0] - bbox_wh[:, 0] / 2.
            tl_y = bbox_cxy[:, 1] - bbox_wh[:, 1] / 2.
            br_x = bbox_cxy[:, 0] + bbox_wh[:, 0] / 2.
            br_y = bbox_cxy[:, 1] + bbox_wh[:, 1] / 2.
            bboxes.append(np.stack([tl_x, tl_y, br_x, br_y], -1))

            # decode keypoints
            per_kps = np.concatenate(
                [((kps_pred[:, [2 * i, 2 * i + 1]] * stride) + anchor_centers)
                 for i in range(self.NK)],
                axis=-1)
            kpss.append(per_kps)

            # final score & labels
            scores.append(cls_score * obj_pred)
            labels.append(cls_label)

        # all branches skipped? → return empty values to avoid crash
        if len(scores) == 0:
            print("[ERROR] No valid outputs due to shape mismatch.")
            return np.zeros((0, 6)), np.zeros((0, self.NK * 2))

        # merge all scales
        scores = np.concatenate(scores, axis=0).reshape(-1)
        bboxes = np.concatenate(bboxes, axis=0)
        kpss = np.concatenate(kpss, axis=0)
        labels = np.concatenate(labels, axis=0).reshape(-1, 1)  # shape: (N, 1)

        # filter scores
        score_mask = (scores > score_thresh)
        scores = scores[score_mask]
        bboxes = bboxes[score_mask]
        kpss = kpss[score_mask]
        labels = labels[score_mask]

        # concatenate label to bbox: x1, y1, x2, y2, score, class_id
        bboxes = np.hstack((bboxes, scores[:, None], labels))

        # print(f"[INFO] Final detections: {len(scores)} faces")
        return bboxes, kpss


    def detect(self, img, score_thresh=0.5, mode='ORIGIN'):
        self.time_engine.tic('preprocess')
        det_img, det_scale = resize_img(img, mode)
        self.time_engine.toc('preprocess')

        # correctly receive two return values
        bboxes, kpss = self.forward(det_img, score_thresh)

        self.time_engine.tic('postprocess')

        # scale bbox and kpss back to original image size
        bboxes[:, :4] /= det_scale
        kpss /= det_scale

        # do NMS: use bboxes[:, :4] for NMS, bboxes[:, 4] for confidence
        pre_det = bboxes[:, :5]  # x1, y1, x2, y2, score
        keep = nms(pre_det, self.nms_thresh)

        # filter
        bboxes = bboxes[keep, :]  # including class_id
        kpss = kpss[keep, :]

        self.time_engine.toc('postprocess')
        return bboxes, kpss


# comment: add function to process single image
def process_single_image(detector, image_path, score_thresh=0.02, mode='640,640'):
    """Process single image detection"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    print(f'Original image size: {img.shape[:-1]}')
    
    # perform detection
    bboxes, kpss = detector.detect(img, score_thresh=score_thresh, mode=mode)
    
    if len(bboxes) > 0:
        print(f'Detected {len(bboxes)} objects')
    else:
        print('No objects detected')
    
    # generate output filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = f"{base_name}_detected.jpg"
    
    # draw detection results
    draw(img, bboxes, kpss, out_path=out_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Object detection using YUNET')
    # comment: simplified command line parameters, keep only necessary ones
    parser.add_argument('image', help='Path to input image')
    parser.add_argument(
        '--mode',
        type=str,
        default='640,640',
        help='Image resize mode, default 640,640')
    parser.add_argument(
        '--score_thresh',
        type=float,
        default=0.02,
        help='Score threshold, default 0.02')
    parser.add_argument(
        '--nms_thresh', 
        type=float, 
        default=0.45, 
        help='NMS threshold, default 0.45')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    # comment: fixed to use YUNET model in current directory
    model_file = './yunet_n_640_640.onnx'
    
    if not os.path.exists(model_file):
        print(f"Error: Model file not found {model_file}")
        print("Please ensure yunet_n_640_640.onnx is in current directory")
        exit(1)
    
    print(f"Loading model: {model_file}")
    detector = YUNET(model_file, nms_thresh=args.nms_thresh)
    
    start = time()
    process_single_image(
        detector,
        args.image,
        score_thresh=args.score_thresh,
        mode=args.mode
    )
    
    end_time = time() - start
    print(f'Total processing time: {end_time:.2f}s')
