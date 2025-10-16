import os
import cv2
import argparse

def resize_image(image_path, output_path, max_size=640):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return

    # 获取当前图片的长和宽
    h, w = img.shape[:2]

    # 根据最大边为640，保持长宽比
    if w > h:
        new_width = max_size
        new_height = int((max_size / w) * h)
    else:
        new_height = max_size
        new_width = int((max_size / h) * w)

    # 调整大小
    resized_img = cv2.resize(img, (new_width, new_height))

    # 保存调整后的图像
    cv2.imwrite(output_path, resized_img)
    print(f"Resized image saved at {output_path}")

def resize_images_in_folder(input_folder, output_folder, max_size=640):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹中所有图片
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            # 为每张图生成一个新的文件路径
            output_path = os.path.join(output_folder, file_name)
            resize_image(file_path, output_path, max_size)

def parse_args():
    parser = argparse.ArgumentParser(description="Resize images in a folder with max edge as 640")
    parser.add_argument('input_folder', help="Path to the input folder containing images")
    parser.add_argument('output_folder', help="Path to the output folder where resized images will be saved")
    parser.add_argument('--max_size', type=int, default=640, help="Maximum size for the longest edge of the image")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    resize_images_in_folder(args.input_folder, args.output_folder, args.max_size)
