import os

import argparse

import json
from tqdm import tqdm

import numpy as np
import cv2

import tensorflow as tf



def enumerate_images(dataset_dir):
    images_list = []
    for dirs, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                images_list.append(os.path.join(dirs, file))

    return images_list


def decode_img(img, height=224, width=224):
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, size=[height, width], method=tf.image.ResizeMethod.BICUBIC)


def process_path(file_path, height=224, width=224):
    img = tf.io.read_file(file_path)
    img = decode_img(img, height=height, width=width)
    return img, file_path


def create_dataset(images_list: list, target_height: int, target_width: int, batch_size: int) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(images_list)
    dataset = dataset.map(lambda x: process_path(x, height=target_height, width=target_width))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def create_model(target_height: int=224, target_width: int=224) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(target_height, target_width, 3))
    model = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(target_height, target_width, 3))
    x = model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.math.l2_normalize(x, axis=1)

    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return model


def images_to_sprite(data):
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)

    data = data.reshape((n, n) + data.shape[1:]).transpose(
        (0, 2, 1, 3) + tuple(range(4, data.ndim + 1))
    )
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


def create_config_file(features_list, images_list, save_dir, single_image_size):
    IMAGES_SPRITE_NAME = "sprite_image.png"
    BASE_DIR = "oss_data"

    features = np.vstack(features_list)
    images = np.stack(images_list, axis=0)

    print("features: {}, images: {}".format(features.shape, images.shape))

    images = images_to_sprite(images)

    if not os.path.exists(os.path.join(save_dir, "oss_data")):
      os.makedirs(os.path.join(save_dir, "oss_data"), exist_ok=True)

    with open(os.path.join(save_dir, "oss_data", "filenames.tsv"), "w+") as f:
        for file_name in files_list:
            f.write(str(file_name) + "\n")

    features.astype(np.float32).tofile(os.path.join(save_dir, "oss_data", "features.bytes"))

    cv2.imwrite(os.path.join(save_dir, "oss_data", IMAGES_SPRITE_NAME), images)

    TENSOR_NAME = "TNSE"
    CONFIG_PATH = "/".join([save_dir, BASE_DIR, "oss_demo_projector_config.json"])
    TENSOR_PATH = "/".join([BASE_DIR, "features.bytes"])
    LABELS_PATH = "/".join([BASE_DIR, "filenames.tsv"])
    IMAGE_SPRITES_PATH = "/".join([BASE_DIR, IMAGES_SPRITE_NAME])
    IMAGE_SIZE = single_image_size


    oss_json = {}

    json_to_append = {
            "tensorPath": TENSOR_PATH,
            "tensorShape": [features.shape[0], features.shape[1]],
            "metadataPath": LABELS_PATH,
            "tensorName": TENSOR_NAME,
            "sprite": {
                "imagePath": IMAGE_SPRITES_PATH,
                "singleImageDim": [IMAGE_SIZE, IMAGE_SIZE]
            }
        }

    oss_json["embeddings"] = [json_to_append]
    oss_json['modelCheckpointPath'] = "Visual Vector Embedding"
    with open(CONFIG_PATH, 'w+') as f:
        json.dump(oss_json, f, indent=4)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default="F:\Datahub_Images")
    parser.add_argument("--target_height", type=int, default=224)
    parser.add_argument("--target_width", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="tnse_tensorboard")
    parser.add_argument("--single_size_image", type=int, default=32)

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = get_args()

    dataset_dir = args.dataset_dir
    target_height = args.target_height
    target_width = args.target_width
    batch_size = args.batch_size
    save_dir = args.save_dir
    single_image_size = args.single_size_image

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    images_list = enumerate_images(dataset_dir) # Liệt kê các ảnh trong thư mục
    print("Num images: {}".format(len(images_list)))
    num_examples = len(images_list)

    model = create_model(target_height=target_height, target_width=target_width) # Tạo model

    dataset = create_dataset(images_list=images_list, target_height=target_height, # Tạo dataset
                             target_width=target_width, batch_size=batch_size)

    """
    Tạo ra list các vector và ảnh
    """
    features_list = []
    images_list = []
    files_list = []
    for image, file_path in tqdm(dataset, total=int(np.ceil(num_examples / batch_size))):
        image_feed = tf.keras.applications.efficientnet.preprocess_input(image)
        features = model.predict(image_feed)
        features_list.append(features)
        file_list = list(map(lambda x: os.path.basename(x.decode("utf-8")), file_path.numpy().tolist()))
        image_list = list(image.numpy())
        assert len(file_list) == len(image_list)
        for img, file_name in zip(image_list, file_list):
            resized_img = cv2.resize(img[:, :, ::-1], (single_image_size, single_image_size))
            images_list.append(resized_img)
            files_list.append(file_name)

    # Tạo config file
    create_config_file(features_list=features_list, images_list=images_list, save_dir=save_dir, single_image_size=single_image_size) 
    