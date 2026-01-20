"""
Bounding box filter code
"""

import os
import numpy as np


def main(labels_path, new_labels_path, score_threshold=0.3, wh_ratio=10.0, wh_threshold=0.01):
    labels_files = os.listdir(labels_path)

    if not os.path.exists(new_labels_path):
        os.makedirs(new_labels_path)

    for i in range(len(labels_files)):
        labels = np.loadtxt(os.path.join(labels_path, labels_files[i]), dtype=np.float32)
        number, _ = labels_files[i].rsplit('.', 1)

        new_labels_ip = os.path.join(new_labels_path, labels_files[i])

        classes = labels[:, 0]
        scores = labels[:, 1]
        boxes = labels[:, 2:]

        new_cls = []
        new_boxes = []
        for cls, score, box in zip(classes, scores, boxes):
            if score > score_threshold:
                new_cls.append(int(cls))
                new_boxes.append(box)

        with open(new_labels_ip, 'w') as f:
            for index, box in enumerate(new_boxes):
                if 1 / wh_ratio < box[2] / box[3] < wh_ratio and box[2] > wh_threshold and box[3] > wh_threshold:
                    f.write(str(new_cls[index]) + ' ')
                    for b in box:
                        f.write(str(b) + ' ')
                    f.write('\n')

                # f.write(str(new_cls[index]) + ' ')
                # for b in box:
                #     f.write(str(b) + ' ')
                # f.write('\n')

        print("{}/{}".format(i + 1, len(labels_files)))


if __name__ == "__main__":
    main(labels_path=' ',
         new_labels_path=' ',
         score_threshold=0.35,
         wh_ratio=1000,
         wh_threshold=0.03)
