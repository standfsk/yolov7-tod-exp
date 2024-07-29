import os
import glob


def visdrone2yolo(dir):
    from PIL import Image
    from tqdm import tqdm
    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    os.makedirs(dir + '/labels', exist_ok=True)  # make labels directory
    pbar = tqdm(glob.glob(os.path.join(dir, 'annotations', '*.txt')), desc=f'Converting {dir}')
    for f in pbar:
        img_size = Image.open(os.path.join(dir, 'images', os.path.basename(f).replace(".txt", ".jpg"))).size
        lines = []
        with open(f, 'r') as file:  # read annotation.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(f).replace(os.sep + 'annotations' + os.sep, os.sep + 'labels' + os.sep), 'w') as fl:
                    fl.writelines(lines)  # write label.txt

for d in ["VisDrone2019-DET-train", "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev"]:
    dpath = os.path.join("datasets", "visDrone", d)
    visdrone2yolo(dpath)