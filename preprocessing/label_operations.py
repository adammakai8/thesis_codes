import xml.etree.ElementTree as ET


class_mapping = {'1db': 0, 'homogen': 1}


def get_label_data(label_path):
    """
    Kiolvassa az xml fájlból a releváns címke adatokat
    :param label_path: Az xml fájl elérési útja
    :return: Egy tömb, ami tömböket tartalmaz a következő formában: [x_center, y_center, width, height, class_id]
    """

    tree = ET.parse(label_path)
    root = tree.getroot()

    size = root.find('size')
    p_width = int(size.find('width').text)
    p_height = int(size.find('height').text)

    bboxes = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id = class_mapping.get(class_name)

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # YOLO formátumú befoglaló téglalap értékek kiszámolása
        width = xmax - xmin
        height = ymax - ymin
        x_center = xmin + (width / 2)
        y_center = ymin + (height / 2)

        bboxes.append([x_center / p_width, y_center / p_height, width / p_width, height / p_height, class_id])

    return bboxes
