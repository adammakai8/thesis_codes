import cv2


RESCALE_SIZE = 640


def resize(image):
    """
    Megviszgálja, hogy a bemeneti kép megfelelő méretű-e, ha nem, akkor átméretezi
    :param image: a bemeneti kép
    :return: A bemeneti kép, ha az megfelelő méretű, egyébként az átméretezett kép
    """
    if image.shape[:2] == [RESCALE_SIZE, RESCALE_SIZE]:
        print('[INFO] A kép a kívánt méretezésnek megfelelő, nincs szükség átméretezésre.')
        return image

    image_scaled = cv2.resize(image, (RESCALE_SIZE, RESCALE_SIZE))

    return image_scaled
