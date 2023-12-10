import os


PATH_SEPARATOR = '/'

DATASET_SUBFOLDERS = ['train', 'test', 'valid']
DATATYPE_SUBFOLDERS = ['images', 'labels']


def create_dir_structure(dataset_name, dataset_path):
    """
    Létrehozza a legenerálandó adathalmaz mappaszerkezetét a paraméterül adott névvel a megadott helyre
    """
    if os.path.exists(dataset_path):
        print('[ERROR] A megadott névvel már létezik ilyen mappa. '
              'Adjon meg másik nevet vagy törölje a névütközést okozó mappát.')
        return False
    print('[INFO] Adathalmaz mappaszerkezetének létrehozása')

    os.mkdir(dataset_path)
    for folder in DATASET_SUBFOLDERS:
        folder_path = f'{dataset_path}{PATH_SEPARATOR}{folder}'
        os.mkdir(folder_path)
        for dtype in DATATYPE_SUBFOLDERS:
            os.mkdir(f'{folder_path}{PATH_SEPARATOR}{dtype}')

    with open(f'{dataset_path}{PATH_SEPARATOR}data.yaml', 'w') as file:
        file.writelines([
            f'path: /content/dataset/{dataset_name}\n',
            'train: train/images\n',
            'val: valid/images\n',
            'test: test/images\n',
            '\n',
            'nc: 2\n',
            'names: [\'1db\', \'homogen\']\n'
        ])

    print('[INFO] Mappaszerkezet és data.yaml létrehozva')
    return True

