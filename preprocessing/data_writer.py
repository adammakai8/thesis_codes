import cv2


PATH_SEPARATOR = '/'


def save_image_and_label(name, image, bboxes, result_path, data_type):
    """ Elmenti a megadott névvel a paraméterül kapott képet és címkéket a megadott helyre """
    if len(bboxes) == 0:
        print(f'[WARNING] A {name} nevű augmentált kép üres címkével rendelkezik, ezért nem mentjük el')
        return

    data_path = f'{result_path}{PATH_SEPARATOR}{data_type}{PATH_SEPARATOR}'
    cv2.imwrite(f'{data_path}images{PATH_SEPARATOR}{name}.jpg', image)
    with open(f'{data_path}labels{PATH_SEPARATOR}{name}.txt', 'w') as file:
        for bbox in bboxes:
            file.write(f'{bbox[4]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')
