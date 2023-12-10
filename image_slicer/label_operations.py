import xml.etree.ElementTree as ET

class_mapping = {'1db': 0, 'homogen': 1}


def get_label_data(label_path):
    """
    Kiolvassa az xml fájlból a releváns címke adatokat
    :param label_path: Az xml fájl elérési útja
    :return: Egy tömb, ami tömböket tartalmaz a következő formában: [xmin, ymin, xmax, ymax, class_id]
    """

    tree = ET.parse(label_path)
    root = tree.getroot()

    bboxes = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id = class_mapping.get(class_name)

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        bboxes.append([xmin, ymin, xmax, ymax, class_id])

    return bboxes


def convert_to_yolov5_format(bboxes, slice_size):
    """
    Befoglaló téglalapok konverziója [xmin, ymin, xmax, ymax, class_id] struktórából YOLO v5 kompatibilis formátumba
    :param slice_size: A kivágott képszelet nagysága (négyzet alakú, ezért egy érték elég)
    :param bboxes: Konvertálandó téglalapok tömbje
    :return: YOLO v5 kompatibilis befoglaló téglalapok tömbje
    """

    result = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, class_id = bbox

        width = xmax - xmin
        height = ymax - ymin
        x_center = xmin + (width / 2)
        y_center = ymin + (height / 2)

        result.append([class_id, x_center / slice_size, y_center / slice_size, width / slice_size, height / slice_size])

    return result
