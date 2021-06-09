"""
The main training file
"""
import os
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import shutil as sh
import fileinput


def parse_start_arguments():
    parser = argparse.ArgumentParser(description='Wheat Detection Training')
    parser.add_argument('--dataset', metavar='S', type=str, required=True,
                        help='Path to the wheat dataset')
    parser.add_argument('--yolo_dir', metavar='S', type=str, required=False,
                        help='Path to the YOLOv5 root')
    parser.add_argument('--log_dir', metavar='S', type=str, default='log',
                        help='Log dir')
    parser.add_argument('--preprocessed', action='store_true',
                        help='Is dataset preprocessed')
    parser.add_argument('--img', metavar='N', type=int, default=1024,
                        help='Image size')
    parser.add_argument('--batch', metavar='N', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--epochs', metavar='N', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--cpu_workers', metavar='N', type=int, default=4,
                        help='Number of cpu workers to process the dataset')
    parser.add_argument('--save_period', metavar='N', type=int, default=10,
                        help='Save model each save_period epochs')
    args = parser.parse_args()
    return args


def get_default_config(args):
    file_path = Path(__file__).parent.absolute()
    yolov5_path = args.yolo_dir if args.yolo_dir else os.path.join(file_path, "yolov5")
    assert os.path.exists(yolov5_path), \
        "Please clone YOLOv5 using command 'git clone https://github.com/ultralytics/yolov5' and provide " \
        "the path to it via --yolo_dir argument"
    assert os.path.exists(args.dataset), \
        f"Dataset does not exist at {args.dataset}"
    log_dir = os.path.join(args.log_dir, datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
    os.makedirs(log_dir, exist_ok=True)

    if args.preprocessed:
        dataset_yaml = os.path.join(args.dataset, "dataset.yaml")
        preprocessed = True
    else:
        dataset_yaml = os.path.join(log_dir, "dataset.yaml")
        preprocessed = False


    config = dict(
        yolov5_dir=yolov5_path,
        dataset_dir=args.dataset,
        train_csv=os.path.join(args.dataset, "train.csv"),
        train_img_dir=os.path.join(args.dataset, "train"),
        log_dir=log_dir,
        dataset_yaml=dataset_yaml,
        dataset_processed=preprocessed,
        image_size=args.img,
        batch=args.batch,
        epochs=args.epochs,
        cpu_workers=args.cpu_workers,
        save_period=args.save_period
    )
    return config


def convert_dataset(config: dict):
    """
    Data in the wheat dataset are given in real coordinates. YOLO expects the relative coordinates in the range [0,1].
    The data are split to rain/val by 80/20 ratio.
    :param config: main training config
    :return: Nothing
    """

    df = pd.read_csv(config["train_csv"])
    bounding_boxes = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        df[column] = bounding_boxes[:, i]
    df.drop(columns=['bbox'], inplace=True)
    df['x_center'] = df['x'] + df['w'] / 2
    df['y_center'] = df['y'] + df['h'] / 2
    df['classes'] = 0

    df = df[['image_id', 'x', 'y', 'w', 'h', 'x_center', 'y_center', 'classes']]

    index = list(set(df.image_id))

    split = 0

    val_index = index[len(index) * split // 5:len(index) * (split + 1) // 5]

    out_dir_train = os.path.join(config["log_dir"], "dataset", "train")
    out_dir_train_images = os.path.join(out_dir_train, "images")
    out_dir_val = os.path.join(config["log_dir"], "dataset", "val")
    out_dir_val_images = os.path.join(out_dir_val, "images")

    for d in (os.path.join(out_dir_train, "labels"), out_dir_train_images,
              os.path.join(out_dir_val, "labels"), out_dir_val_images):
        os.makedirs(d)

    for name, mini in tqdm(df.groupby('image_id')):
        output_dir = out_dir_val if name in val_index else out_dir_train
        label_filename = os.path.join(output_dir, "labels", f"{name}.txt")
        images_dir = os.path.join(output_dir, "images")

        with open(label_filename, 'w+') as f:
            row = mini[['classes', 'x_center', 'y_center', 'w', 'h']].astype(float).values
            row = row / 1024  # we know the exact image size in the wheat dataset
            row = row.astype(str)
            for j in range(len(row)):
                text = ' '.join(row[j])
                f.write(text)
                f.write("\n")
        # Copy images
        original_image_path = os.path.join(config["train_img_dir"], f"{name}.jpg")
        sh.copy(original_image_path, os.path.join(images_dir, f"{name}.jpg"))

    # Save yaml file containing dataset data to be used during training
    with open(config["dataset_yaml"], "w+") as f:
        f.write(f"train: {os.path.abspath(out_dir_train_images)}\n")
        f.write(f"val: {os.path.abspath(out_dir_val_images)}\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['wheat']\n")


def rewrite_yolo_train():
    """
    Changes the train.py of YOLOv5 so it can be run from a python script directly
    :return:
    """
    orig_filename = os.path.join(config["yolov5_dir"], "train.py")
    new_filename = os.path.join(config["yolov5_dir"], "train_fixed.py")


    sh.copy(orig_filename, new_filename)

    with open(new_filename, 'r') as file:
        filedata = file.read()

    # Replace the target strings
    main_replacement_text = "if __name__ == '__main__':\n    import sys\n    main(sys.argv[1:])\n"
    filedata = filedata.replace("if __name__ == '__main__':", "def main(args):")
    # filedata = filedata.replace("parser = argparse.ArgumentParser()", "parser = argparse.ArgumentParser(args)")
    filedata = filedata.replace("opt = parser.parse_args()", "opt = parser.parse_args(args)")


    # Write the file out again
    with open(new_filename, 'w') as file:
        file.write(filedata)

    with open(new_filename, "a+") as f:
        f.write("\n\n")
        f.write(main_replacement_text)


if __name__ == "__main__":
    args = parse_start_arguments()
    config = get_default_config(args)
    if not config["dataset_processed"]:
        convert_dataset(config)

    rewrite_yolo_train()

    import yolov5.train_fixed as yolo_train
    yolo_train.main(["--img", config["image_size"],
                     "--batch", config["batch"],
                     "--epochs", config["epochs"],
                     "--data", config["dataset_yaml"],
                     "--cfg", os.path.join(config['yolov5_dir'], "models", "yolov5s.yaml"),
                     "--project", "yolov5s_wheat",
                     "--device", "0",
                     "--adam",
                     "--workers", config["cpu_workers"],
                     "--name", config["log_dir"],
                     "--save_period", config["save_period"],
                     "--single-cls"])

    print("FINISHED")
