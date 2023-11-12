import os
from PIL import Image


def crop_and_save_right_half(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹内的所有图片文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

    for image_file in image_files:
        input_image_path = os.path.join(input_folder, image_file)
        output_image_path = os.path.join(output_folder, image_file)

        # 打开图像
        image = Image.open(input_image_path)

        # 获取图像的宽度和高度
        width, height = image.size

        # 计算分割位置
        split_position = width // 2

        # 裁剪右半部分
        right_half = image.crop((split_position, 0, width, height))

        # 保存裁剪后的图像
        right_half.save(output_image_path)


# 输入和输出文件夹路径
input_folder = 'input/style-ori'  # 替换为你的输入文件夹路径
output_folder = 'input/sty'  # 替换为输出文件夹路径

# 调用函数进行处理
crop_and_save_right_half(input_folder, output_folder)