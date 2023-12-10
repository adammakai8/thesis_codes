import os


DETECT_DIR = '../runs/detect'
RUN_DIR = f'{DETECT_DIR}/mo_reszlet_transpose_conf02'


def F1_score(tp, fp, fn):
    """
    F1 értéket számolja ki
    :param tp: Igaz pozitív elemek száma
    :param fp: Hamis pozitív elemek száma
    :param fn: Hamis negatív elemek száma
    :return: F1 érték
    """
    return (2 * tp) / (2 * tp + fp + fn)


datasets = os.listdir(RUN_DIR)

negative = True
true_positive = 0
false_positive = 0
false_negative = 0

for dataset in datasets:
    data_count = len(os.listdir(f'{RUN_DIR}/{dataset}')) - 2
    labels = os.listdir(f'{RUN_DIR}/{dataset}/labels')

    detected = len(labels)
    solo = 0
    homogen = 0
    both = 0

    for label in labels:
        with open(f'{RUN_DIR}/{dataset}/labels/{label}', 'r') as file:
            lines = file.readlines()
            classes = [line.split(' ')[0] for line in lines]
            if '0' in classes:
                solo += 1
            if '1' in classes:
                homogen += 1
            if '0' in classes and '1' in classes:
                both += 1

    negative = not negative

    if negative:
        false_positive = detected
    else:
        true_positive = detected
        false_negative = data_count - detected

    with open(f'{RUN_DIR}/summary.txt', 'a') as output:
        output.write(f'{dataset}\n')
        output.write(f'Összes adat száma: {data_count}\n')
        output.write(f'Ezekből pozitívnak detektáltak: {detected} ({round(detected/data_count * 100, 2)} %)\n')
        output.write(f'Solo növény detektálva: {solo} ({round(solo/data_count * 100, 2)} %)\n')
        output.write(f'Homogén terület detektálva: {homogen} ({round(homogen/data_count * 100, 2)} %)\n')
        output.write(f'Mindkét osztályból detektálva: {both} ({round(both/data_count * 100, 2)} %)\n')
        output.write('\n')

        if negative:
            output.write(f'{dataset[:len(dataset) - 1]} F1 értéke: {F1_score(true_positive, false_positive, false_negative)}\n')
            output.write('\n')
