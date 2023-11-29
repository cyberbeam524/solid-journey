import argparse
import glob
import importlib
import json
import matplotlib
import os
import requests
import shutil
import time
import torch
import yaml

from pathlib import Path

from dagshub.streaming import install_hooks
install_hooks()


def download_training_scripts(outpath):
    train = 'https://raw.githubusercontent.com/ultralytics/yolov5/master/train.py'
    val = 'https://raw.githubusercontent.com/ultralytics/yolov5/master/val.py'

    for url in (train, val):
        os.makedirs(outpath, exist_ok=True)

        script = os.path.join(outpath, os.path.split(url)[-1])
        if os.path.exists(script):
            continue

        res = requests.get(url)

        if type(res.content) is bytes:
            mode = 'wb'
        else:
            mode = 'w'

        with open(script, mode='wb') as f:
            f.write(res.content)


def read_yaml(filename):
    with open(filename) as f:
        return yaml.safe_load(f)


def save_yaml(filename, data):
    with open(filename, mode='w') as f:
        f.write(yaml.safe_dump(data))


def custom_img2label_paths(img_paths):
    import os
    paths, img_names = zip(*[os.path.split(i) for i in img_paths])
    paths = [os.path.join(p.rsplit('/data/', 1)[0], 'annotations/labels') for p in paths]
    ann_names = [os.path.splitext(i)[0] + '.txt' for i in img_names]
    return [os.path.join(p, a) for p, a in zip(paths, ann_names)]
        

def get_labeled_images():
    labeled_imgs = set()

    labelstudio_files = glob.glob('../.labelstudio/*.json')
    for ls_file in labelstudio_files:
        with open(ls_file) as f:
            annotations = json.load(f)

        # If there's only one annotations, the Label Studio JSON file uses a `dict` as the top
        # level structure. However, it will use a `list`, if there are multiple. To make processing
        # easier, convert `dict`s to `list`s.
        if type(annotations) is dict:
            annotations = [annotations]

        for annotation in annotations:
            img = annotation['data']['image']
            _, img = os.path.split(img)
            labeled_imgs.add(img)

    return labeled_imgs


def main():
    parser = argparse.ArgumentParser("Train a YOLOv5 model to detect squirrels")
    parser.add_argument("--data", required=True, help="Path to YAML file describing the data to use for training, validation, and testing")
    parser.add_argument("--weights", required=True, help="Path to the pretrained YOLOv5 weights to use")
    parser.add_argument("--epochs", default=300, type=int, help="Number of epochs to run")
    parser.add_argument("--batch-size", default=16, type=int, help="Batch size to use for training")
    parser.add_argument("--save-path", required=True, help="Path to save the best training results to")

    args = parser.parse_args()

    temp_yolov5_path = 'yolov5'

    download_training_scripts(temp_yolov5_path)

    # This will also locally cache the YOLOv5 repo
    _ = torch.hub.load('ultralytics/yolov5', 'custom', path=args.weights)

    import utils

    utils.dataloaders.img2label_paths = custom_img2label_paths
    orig_cache_labels = utils.dataloaders.LoadImagesAndLabels.cache_labels

    labeled_imgs = get_labeled_images()

    def custom_cache_labels(self, path=Path('./labels.cache'), prefix=''):
        data_cnt = len(self.im_files)
        for i in reversed(range(data_cnt)):
            im_file = self.im_files[i]
            _, im_name = os.path.split(im_file)
            if im_name not in labeled_imgs:
                self.im_files = self.im_files[:i] + self.im_files[i+1:]
                self.label_files = self.label_files[:i] + self.label_files[i+1:]

        return orig_cache_labels(self, path, prefix)

    utils.dataloaders.LoadImagesAndLabels.cache_labels = custom_cache_labels

    train = importlib.import_module(f'{temp_yolov5_path}.train')

    data_yaml = read_yaml(args.data)
    yaml_path, yaml_name =os.path.split(os.path.abspath(args.data))
    data_yaml['path'] = os.path.join(yaml_path, data_yaml['path'])

    new_yaml = os.path.join(temp_yolov5_path, yaml_name)

    save_yaml(new_yaml, data_yaml)

    train.run(weights=args.weights, data=new_yaml, hyp='data/hyps/hyp.scratch-low.yaml', epochs=args.epochs, batch_size=args.batch_size, name='squirrel', exist_ok=True)

    best_weights = f'{temp_yolov5_path}/runs/train/squirrel/weights/best.pt'
    outpath = args.save_path
    if outpath.endswith('.pt'):
        outdir, outfile = os.path.split(outpath)
    else:
        outdir = outpath
        outfile = f'{str(time.time_ns())[:-6]}.pt'
        outpath = os.path.join(outdir, outfile)

    if os.path.exists(best_weights):
        os.makedirs(outdir, exist_ok=True)
        try:
            shutil.copy2(best_weights, outpath)
        except PermissionError:
            print(f"Permission denied when trying to save model to '{outpath}'")
        except:
            print(f"Error occurred while trying to save model to '{outpath}'")
        else:
            shutil.rmtree(f'{temp_yolov5_path}/runs', ignore_errors=True)


if __name__ == '__main__':
    main()
