import argparse
import os
from gdino import GroundingDINOAPIWrapper, visualize
from PIL import Image
import cv2
import torch
from torchvision.ops import box_convert


def get_args():
    parser = argparse.ArgumentParser(description="Interactive Inference")
    parser.add_argument(
        "--token",
        default=' ',  # The official token needs to be purchased from the official website
        type=str,
        help="The token for T-Rex2 API. We are now opening free API access to T-Rex2",
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.25, help="The threshold for box score"
    )
    return parser.parse_args()


image_path = ' '  # Image folder address
image_files = os.listdir(image_path)

new_label_path = image_path.replace('image', 'dino1.5_labels')
new_image_path = image_path.replace('label', 'dino1.5_images')
if not os.path.exists(new_label_path):
    os.makedirs(new_label_path)
if not os.path.exists(new_image_path):
    os.makedirs(new_image_path)

args = get_args()
gdino = GroundingDINOAPIWrapper(args.token)

for i in range(len(image_files)):
    img_path = image_path + '/' + image_files[i]
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    number, _ = image_files[i].rsplit('.', 1)
    dino_label_ip = new_label_path + '/' + number + '.txt'

    prompts = dict(image=img_path, prompt='fruit')
    results = gdino.inference(prompts)
    boxes = results["boxes"]
    scores = results["scores"]
    categorys = results["categorys"]

    new_boxes = []
    new_scores = []
    new_categorys = []
    for box, score, category in zip(boxes, scores, categorys):
        box = torch.Tensor(box)
        box = box_convert(box, in_fmt='xyxy', out_fmt='cxcywh').numpy()
        box[0] /= w
        box[2] /= w
        box[1] /= h
        box[3] /= h
        # if 0.5 < box[2] / box[3] < 2 and box[2] > 0.04 and box[3] > 0.04 and score > 0.42:
        if score > 0.25:
            new_boxes.append(box)
            new_scores.append(score)
            new_categorys.append(category)

    with open(dino_label_ip, 'w') as f:
        for j, box in enumerate(new_boxes):
            f.write('0 ')
            f.write(str(new_scores[j]) + " ")
            for b in box:
                f.write(str(b) + ' ')
            f.write('\n')

    # now visualize the results
    image_pil = Image.open(prompts['image'])
    image_pil = visualize(image_pil, results)
    # dump the image to the disk
    image_pil.save(new_image_path + '/' + image_files[i])
    print("已完成{}/{}".format(i + 1, len(image_files)))
