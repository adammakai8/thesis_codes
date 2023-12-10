import random
import cv2
import os

import label_operations as lop
import augmentations as aug
import directories as dirs
import data_writer as dw
import resize as r


PATH_SEPARATOR = '/'

TEST_INSTANCES_PER_YEAR = 20
VAL_INSTANCES_PER_YEAR = 20

DATASET_NAME = '8x200_std'
RESULT_PATH = f'../datasets/{DATASET_NAME}'

DATA_PATH = '../datasets/8x200'


def read_data_from_paths(image_path, label_path):
    return cv2.imread(image_path), lop.get_label_data(label_path)


def get_item_by_substring(sub_str, collection):
    """
    Megadott substring alapján kikeresi és visszaadja a kollekcióból az első elemet, amiben megtalálható a substring
    """
    result = [element for element in collection if sub_str in element]
    if len(result) > 0:
        return result[0]
    else:
        return None


def group_paths(names, image_paths, label_paths):
    """
    Összerendeli dictionary-be az adatpélda nevét (ez a kulcs) a kép és a címkefájl elérési útjával
    """
    result = {}
    for name in names:
        result[name] = {
            'image_path': f'{DATA_PATH}{PATH_SEPARATOR}{get_item_by_substring(name, image_paths)}',
            'label_path': f'{DATA_PATH}{PATH_SEPARATOR}{get_item_by_substring(name, label_paths)}'
        }
    return result


def sort_train_test_val(names, paths):
    """
    Szétválogatja évenként a példákat tanító, teszt és validációs halmazokra.
    A teszt és validációs halmazokat elmenti, a tanító halmaz elemeinek neveit visszaadja.

    :param names: példák neve (fájlnév kiterjesztés nélkül)
    :param paths: az adatpéldák képeinek és címkéinek elérési útjai
    :return: A train adathalmazba kiválogatott példák nevei
    """
    print('[INFO] Adatpéldák szétválogatása')
    names_grouped = [
        list(filter(lambda name: name[:4] == '2009', names)),
        list(filter(lambda name: name[:4] == '2012', names)),
        list(filter(lambda name: name[:4] == '2015', names)),
        list(filter(lambda name: name[:4] == '2018', names))
    ]

    for group in names_grouped:
        if len(group) < TEST_INSTANCES_PER_YEAR + VAL_INSTANCES_PER_YEAR:
            continue

        picked = random.sample(group, TEST_INSTANCES_PER_YEAR + VAL_INSTANCES_PER_YEAR)
        test_set = picked[:TEST_INSTANCES_PER_YEAR]
        val_set = picked[TEST_INSTANCES_PER_YEAR:TEST_INSTANCES_PER_YEAR + VAL_INSTANCES_PER_YEAR]

        # Teszt és validációs adatok skálázása, majd mentése
        for item in test_set:
            if 'None' in paths[item]['label_path']:
                print(f'[ERROR] A(z) {item} képhez nem társítható címkefájl, adathalmazból kihagyva')
                continue

            image, bboxes = read_data_from_paths(paths[item]['image_path'], paths[item]['label_path'])
            image_scaled = r.resize(image)
            dw.save_image_and_label(item, image_scaled, bboxes, RESULT_PATH, 'test')

        for item in val_set:
            if 'None' in paths[item]['label_path']:
                print(f'[ERROR] A(z) {item} képhez nem társítható címkefájl, adathalmazból kihagyva')
                continue
            image, bboxes = read_data_from_paths(paths[item]['image_path'], paths[item]['label_path'])
            image_scaled = r.resize(image)
            dw.save_image_and_label(item, image_scaled, bboxes, RESULT_PATH, 'valid')

        # Ezek az adatok nem kerülnek bele a tanító halmazba, ezért a visszaadandó listából kitöröljük őket
        for item in picked:
            names.remove(item)

    print('[INFO] Adatpéldák szétválogatva')
    return names


def preprocess_and_augment_trainset(names, paths):
    """ A tanítóhalmaz előfeldolgozását és augmentációját végző függvény """
    print('[INFO] Tanítópéldák augmentációja')
    i = 1
    for name in names:
        print(f'[INFO] {i}/{len(names)} kép augmentációja')
        if 'None' in paths[name]['label_path']:
            print(f'[ERROR] {i}. képhez nem társítható címkefájl, augmentációból kihagyva')
            continue

        image, bboxes = read_data_from_paths(paths[name]['image_path'], paths[name]['label_path'])
        image_scaled = r.resize(image)
        aug.augment_image_and_label(image_scaled, bboxes, name, RESULT_PATH, aug.MODES.REPEATED, 5)
        i += 1
    print('[INFO] Képek augmentációja kész')


def preprocess_images_and_labels():
    """ Az előfeldolgozás és augmentációs folyamat belépési pontja """

    if not dirs.create_dir_structure(DATASET_NAME, RESULT_PATH):
        print('[INFO] A folyamat meghiúsult, kilépés...')
        return

    print('[INFO] Fájlok keresése és rendszerezése')
    file_paths = os.listdir(DATA_PATH)
    image_paths = list(filter(lambda filename: filename.split('.')[-1] == 'jpg', file_paths))
    label_paths = list(filter(lambda filename: filename.split('.')[-1] == 'xml', file_paths))
    names = [path.split(PATH_SEPARATOR)[-1].split('.')[0] for path in image_paths]
    paths_grouped = group_paths(names, image_paths, label_paths)
    print('[INFO] Keresés és rendszerezés kész')

    train_names = sort_train_test_val(names, paths_grouped)
    preprocess_and_augment_trainset(train_names, paths_grouped)


preprocess_images_and_labels()
