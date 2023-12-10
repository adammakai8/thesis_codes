import cv2
import numpy as np
from itertools import product

import label_operations
from aop_logger import log_runtime_in_debug


SLICE_SIZE = 640
THRESHOLD_SIZE = 4096
RULE_OUT_RADIUS = 64

TILE_NUM = 6


@log_runtime_in_debug
def create_mask(image, boxes):
    """
    Létrehozza a maszkot, ami megmutatja, hol lehetnek a potenciális szeletek bal felső sarkai (xmin, ymin)
    :param image: A kép, amire a maszk készül
    :param boxes: A címkék befoglaló téglalapjai
    :return: Maszk, ahol 1-es potenciális kiindulási sarokpont a 0 pedig a többi
    """

    height, width = image.shape[:2]
    mask = np.zeros([height, width])

    # Adjuk hozzá a maszkhoz azokat a régiókat amik biztosan átfedésben vannak legalább egy címkével
    for box_any in boxes:
        xmax, ymax = box_any[2:4]
        mask[max(ymax - SLICE_SIZE + 1, 0):ymax, max(xmax - SLICE_SIZE + 1, 0): xmax] = 1

    # Vegyük ki azt a régiót, ahol a szelet már lelógna a képről
    mask[height - SLICE_SIZE:, :] = 0
    mask[:, width - SLICE_SIZE:] = 0

    # Az 1db címkéket nem lehet ketté vágni, azokat a régiókat, ahol ez megtörténne szintén vegyük ki
    boxes_1db = list(filter(lambda box: box[4] == 0, boxes))
    for box_1db in boxes_1db:
        xmin, ymin, xmax, ymax = box_1db[:4]
        mask[max(ymin - SLICE_SIZE, 0):max(ymax - SLICE_SIZE + 1, 0), max(xmin - SLICE_SIZE, 0):xmax + 1] = 0
        mask[ymin:ymax + 1, max(xmin - SLICE_SIZE, 0):xmax + 1] = 0
        mask[max(ymax - SLICE_SIZE, 0):ymin + 1, max(xmin - SLICE_SIZE, 0):max(xmax - SLICE_SIZE + 1, 0)] = 0
        mask[max(ymax - SLICE_SIZE, 0): ymin + 1, xmin:xmax + 1] = 0

    return mask


def adjust_bbox_coordinates(bbox, xl, yl, xu, yu):
    """
    A befoglaló téglalap koordinátáit a kivágandó képszelethez igazítja, amennyiben
    a téglalap benne van a szeletben és homogén címke esetén területe meghaladja a megadott korlátot
    :param bbox: A vizsgálandó befoglaló téglalap
    :param xl: A szelet alsó x koordinátája
    :param yl: A szelet alsó y koordinátája
    :param xu: A szelet felső x koordinátája
    :param yu: A szelet felső y koordinátája
    :return: Befoglaló téglalap a kiigazított koordinátákkal, vagy None, ha nem felel meg a feltételeknek
    """

    if xl < bbox[2] and yl < bbox[3] and xu > bbox[0] and yu > bbox[1]:
        xmin = bbox[0] - xl if bbox[0] > xl else 0
        ymin = bbox[1] - yl if bbox[1] > yl else 0
        xmax = bbox[2] - xl if bbox[2] < xu else 640
        ymax = bbox[3] - yl if bbox[3] < yu else 640

        if bbox[4] == 0 or (bbox[4] == 1 and (xmax - xmin) * (ymax - ymin) > THRESHOLD_SIZE):
            return [xmin, ymin, xmax, ymax, bbox[4]]

    return None


def get_boxes_from_slice(x, y, bndboxes):
    """
    A képszeletben található befoglaló téglalapokat adja vissza
    :param x: A szelet bal felső x koordinátája
    :param y: A szelet bal felső y koordinátája
    :param bndboxes: A képen található összes befoglaló téglalap
    :return: Egy tömb, ami a kivágandó szeletben található téglalapokat tartalmazza
    """

    sliced_boxes = []
    for box in bndboxes:
        adjusted = adjust_bbox_coordinates(box, x, y, x + SLICE_SIZE, y + SLICE_SIZE)
        if adjusted is not None:
            sliced_boxes.append(adjusted)
    return sliced_boxes


@log_runtime_in_debug
def save_sliced_data(slice_num, image, labels, data_name, dest_path):
    """
    A kivágott képszelet és a hozzátartozó címkéket menti
    :param slice_num: A szelet sorszáma
    :param image: Kivágott képszelet
    :param labels: Címkék és befoglaló téglalapok
    :param data_name: Az eredeti adatpélda neve
    :param dest_path: Az eredmény adatok mapparendszere
    """

    dest_img = f'{dest_path}/images'
    dest_label = f'{dest_path}/labels'

    cv2.imwrite(f'{dest_img}/{data_name}_{slice_num}.jpg', image)

    yolo_labels = label_operations.convert_to_yolov5_format(labels, SLICE_SIZE)
    with open(f'{dest_label}/{data_name}_{slice_num}.txt', 'w') as file:
        for lbl in yolo_labels:
            class_id, x_center, y_center, width, height = lbl
            file.write(f'{class_id} {x_center} {y_center} {width} {height}\n')


@log_runtime_in_debug
def make_slices(image_src, label_src, data_name, dest_path):
    """
    A megadott elérések alapján beolvassa a képet és címkéit, felszeleteli a képet,
    hogy minden szeletben legyen címke és az 1db címkéket ne vágja ketté,
    majd elmenti a sikeresen szeletelt képeket és a hozzájuk igazított címkéket
    :param image_src: A képfájl neve
    :param label_src: A címkefájl neve
    :param data_name: Az adatpélda neve
    :param dest_path: A kimeneti mapparendszer neve
    :return: A sikeresen kivágott képszeletek száma
    """

    image = cv2.imread(image_src)
    bndboxes = label_operations.get_label_data(label_src)

    mask = create_mask(image, bndboxes)

    # Maszk felosztása kisebb szeletekre, hogy mindig más régióból vágjon ki képet, így nem lesznek a szeletek hasonlók
    horizontal_tile_indexes = np.linspace(0, mask.shape[0], TILE_NUM + 1, dtype=np.int32)[:-1]
    vertical_tile_indexes = np.linspace(0, mask.shape[1], TILE_NUM + 1, dtype=np.int32)[:-1]

    num = 0
    for h_tile, v_tile in product(horizontal_tile_indexes, vertical_tile_indexes):
        # A maszkból kiszedjük az aktuális szeletet, ott választunk ki egy kivágandó képszeletet
        submask = np.zeros(mask.shape)
        submask[h_tile:h_tile + int(mask.shape[0] / TILE_NUM), v_tile:v_tile + int(mask.shape[1] / TILE_NUM)] = 1
        submask *= mask

        while submask.sum() > 0:
            coordinates = np.argwhere(submask == 1)
            y, x = coordinates[np.random.choice(coordinates.shape[0])]
            sliced_bboxes = get_boxes_from_slice(x, y, bndboxes)
            if len(sliced_bboxes) > 0:
                sliced_image = image[y:y + SLICE_SIZE, x:x + SLICE_SIZE]
                save_sliced_data(num, sliced_image, sliced_bboxes, data_name, dest_path)

                # A kiválasztott pont környezetét kivesszük a további lehetőségek közül, hogy ne kapjunk hasonló képeket
                mask[max(y - RULE_OUT_RADIUS, 0):min(y + RULE_OUT_RADIUS, mask.shape[0]), max(x - RULE_OUT_RADIUS, 0):min(x + RULE_OUT_RADIUS, mask.shape[1])] = 0
                submask = np.zeros(mask.shape)

                num += 1
            else:
                mini_radius = int(RULE_OUT_RADIUS / 4)
                submask[max(y - mini_radius, 0):min(y + mini_radius, mask.shape[0]), max(x - mini_radius, 0):min(x + mini_radius, mask.shape[1])] = 0

    return num
