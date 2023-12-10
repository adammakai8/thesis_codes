import enum
import random as rng

import albumentations as A
import data_writer as dw


class MODES(enum.Enum):
    """ Az augmentálás módját adja meg """
    SEQUENTIAL = 1, "A beállított augmentációs módszerekhez egy-egy képet generál"
    RANDOM = 2, "A megadott augmentációkat véletlenszerűen alkalmazza a képekre"
    REPEATED = 3, "A megadott augmentációk közül mindet alkalmazza minden képre, adott ismétlődéssel"


CROP_PARAMS = []

BBOX_PARAMS = A.BboxParams(format='yolo', min_visibility=0.1)

TRANSFORMS_SEQ = [
    A.Compose([A.HorizontalFlip(p=1), A.Blur(5, p=0.5)], bbox_params=BBOX_PARAMS),
    A.Compose([A.Rotate(limit=(-30, 30), p=1), A.ISONoise((0.02, 0.03), (0.2, 0.3), p=0.5)], bbox_params=BBOX_PARAMS),
    A.Compose([A.HorizontalFlip(p=1), A.Rotate(limit=(-30, 30), p=1), A.ISONoise((0.02, 0.03), (0.2, 0.3), p=0.5)],
              bbox_params=BBOX_PARAMS),
    A.Compose([A.Affine(shear=(15, 15), p=1), A.Blur(3, p=0.5)], bbox_params=BBOX_PARAMS),
    A.Compose([A.HorizontalFlip(p=1), A.Affine(shear=(15, 15), p=1), A.Blur(3, p=0.5)], bbox_params=BBOX_PARAMS),
    A.Compose([A.Cutout(p=1)], bbox_params=BBOX_PARAMS),
    A.Compose([A.HorizontalFlip(p=1), A.Cutout(p=1), A.ISONoise((0.02, 0.03), (0.2, 0.3), p=0.5)],
              bbox_params=BBOX_PARAMS),
]

TRANSFORMS_RND = A.Compose([
    A.HorizontalFlip(),
    A.RandomResizedCrop(640, 640, (0.6, 0.9), (1, 1)),
    A.Blur(),
    A.ISONoise(),
    A.Cutout(6, 60, 60),
    A.Rotate(limit=(-30, 30)),
    A.Affine(shear=(-15, 15)),
], bbox_params=BBOX_PARAMS)


def get_transform_rep():
    return A.Compose([
            A.ColorJitter(p=1),
            A.GaussianBlur(p=1),
            A.Cutout(num_holes=5, max_h_size=rng.randint(1, 64), max_w_size=rng.randint(1, 64), p=1)
        ], bbox_params=BBOX_PARAMS)


def augment_image_and_label(image, bboxes, name, result_path, mode=MODES.RANDOM, amount=2):
    """
    Különböző augmentációs technikák alkalmazásával generál új tanítópéldákat
    :param image: Az eredeti kép
    :param bboxes: Az eredeti kép címkéi
    :param name: A tanítópélda neve
    :param result_path: Az adathalmaz mentési helye
    :param mode: Az augmentálás módja (lásd :class:`MODES`)
    :param amount: Hányszorosára akarjuk növelni az adathalmazt (:class:`MODES.SEQUENTIAL` esetén nem használjuk)
    """

    # Eredeti képet is elmentjük
    dw.save_image_and_label(f'{name}_0', image, bboxes, result_path, 'train')

    if mode == MODES.RANDOM:
        for i in range(1, amount):
            transformed = TRANSFORMS_RND(image=image, bboxes=bboxes)
            dw.save_image_and_label(f'{name}_{i}', transformed['image'], transformed['bboxes'], result_path, 'train')
    elif mode == MODES.SEQUENTIAL:
        i = 1
        # Transzformációkon végigmegyünk és egyesével elvégezzük a képre, majd a transzformált képeket elmentjük
        for transform in TRANSFORMS_SEQ:
            transformed = transform(image=image, bboxes=bboxes)
            dw.save_image_and_label(f'{name}_{i}', transformed['image'], transformed['bboxes'], result_path, 'train')
            i += 1
    else:
        for i in range(1, amount):
            transformed = get_transform_rep()(image=image, bboxes=bboxes)
            dw.save_image_and_label(f'{name}_{i}', transformed['image'], transformed['bboxes'], result_path, 'train')
