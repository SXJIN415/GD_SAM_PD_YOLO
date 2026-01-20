import os
import sys

import shutil
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def seg_with_sam_with_box_prompt(image, boxes, predictor, show_img=False):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        input_boxes = boxes.astype(int)
        masks, scores, _ = predictor.predict(point_coords=None,
                                             point_labels=None,
                                             box=input_boxes,
                                             multimask_output=False)

        if show_img:
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.squeeze(0), plt.gca(), random_color=True)
            for box in input_boxes:
                show_box(box, plt.gca())
            plt.axis('off')
            plt.show()
        return masks, scores


def mask2txt(masks, w, h):
    yolo_label = []

    for i in range(masks.shape[0]):
        mask = masks[i].squeeze()
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        new_contours = []
        for index, contour in enumerate(contours):
            contour = np.array(contour, dtype=np.float32).squeeze(1)
            if index == 0:
                new_contours = contour
            else:
                new_contours = np.concatenate((new_contours, contour), axis=0)
        new_contours[:, 0] = new_contours[:, 0] / w
        new_contours[:, 1] = new_contours[:, 1] / h
        new_contours = np.array(np.array(new_contours).flatten())

        yolo_label.append(new_contours)

    return yolo_label


def get_masks(image_files_path, label_files_path, save_label_path, single=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sam_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_model = build_sam2(model_cfg, sam_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam_model)

    image_files_name = os.listdir(image_files_path)
    label_files_name = os.listdir(label_files_path)
    if not os.path.exists(save_label_path):
        os.makedirs(save_label_path)

    # for i in range(len(image_files_name)):
    #     image_number, _ = image_files_name[i].rsplit('.', 1)
    #     label_number, _ = label_files_name[i].rsplit('.', 1)
    #     if image_number != label_number:
    #         print('标签与图片不符，退出程序！')
    #         sys.exit(1)

    for i in range(len(image_files_name)):
        image = cv2.imread(image_files_path + '/' + image_files_name[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = np.loadtxt(label_files_path + '/' + label_files_name[i])
        if len(label.shape) == 1:
            label = label[np.newaxis, :]

        h, w, _ = image.shape
        boxes = label[:, 1:]
        classes = label[:, 0]
        x, y, b_x, b_h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (x - b_x / 2) * w
        y1 = (y - b_h / 2) * h
        x2 = (x + b_x / 2) * w
        y2 = (y + b_h / 2) * h
        boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = x1, y1, x2, y2

        masks, scores = seg_with_sam_with_box_prompt(image, boxes, predictor)

        seg_label = mask2txt(masks, w, h)

        with open(os.path.join(save_label_path, label_files_name[i]), "w") as f:
            for index, line in enumerate(seg_label):
                if single:
                    f.write('0 ')
                else:
                    f.write(str(int(classes[index])) + ' ')
                # f.write(str(line.round(6)).strip('[]'))
                for j in range(len(line)):
                    one = str(line[j])
                    f.write(one + ' ')
                f.write(str(float(scores[index])))
                f.write('\n')

        print('{}/{}'.format(i + 1, len(image_files_name)))
    # 在label中保存classes.txt文件
    # destination_file = os.path.join(save_label_path, label_files_name[-1])
    # shutil.copyfile(os.path.join(label_files_path, label_files_name[-1]), destination_file)
    print('程序已完成！')


if __name__ == '__main__':
    get_masks(image_files_path='',  # Picture folder address
              label_files_path='',  # Bounding box folder address
              save_label_path='',  # Mask folder address
              single=True
              )
