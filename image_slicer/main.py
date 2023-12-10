import os
import slicer

DATASET_NAME = '8x200'
SOURCE_PATH = f'../../datasets/{DATASET_NAME}'
DESTINATION_PATH = f'../sliced_data/v2_{DATASET_NAME}_slices'


def make_folder():
    """ Létrehozza az eredményül kapott szeletek eltárolására használt mapparendszert """

    if os.path.exists(DESTINATION_PATH):
        print('[ERROR] A megadott névvel már létezik ilyen mappa. '
              'Adjon meg másik nevet vagy törölje a névütközést okozó mappát.')
        return False
    os.mkdir(DESTINATION_PATH)
    os.mkdir(f'{DESTINATION_PATH}/images')
    os.mkdir(f'{DESTINATION_PATH}/labels')
    print('[INFO] Eredmény könyvtárak létrehozva')


def read_data_and_slice():
    """
    A képek és címkék beolvasásáért, illetve a szeletek elkészítéséért felelő funkciókat fogja össze
    és elvégzi az összes kiindulási adatra
    """

    print('[INFO] Képek szeletelése:')
    img_names = list(filter(lambda filename: filename.split('.')[-1] == 'jpg', os.listdir(SOURCE_PATH)))
    i = 1
    for img_name in img_names:
        print(f'[INFO] {i}/{len(img_names)} kép feldolgozása... ', end='')
        label_name = f'{img_name[:len(img_name) - 3]}xml'
        if os.path.exists(f'{SOURCE_PATH}/{label_name}'):
            img_src = f'{SOURCE_PATH}/{img_name}'
            lbl_src = f'{SOURCE_PATH}/{label_name}'

            data_name = img_name[:len(img_name) - 3]
            num = slicer.make_slices(img_src, lbl_src, data_name, DESTINATION_PATH)
            print(f'Felszeletelt képek száma: {num}')
        else:
            print(f'\n [ERROR] A(z) {img_name} képhez nem társítható címkefájl, adathalmazból kihagyva')
        i += 1


if __name__ == '__main__':
    make_folder()
    read_data_and_slice()
    print('Képek feldolgozása kész.')
