import csv

def load_disease_map(csv_path):
    disease_map = {}

    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            patient_id = row[0].strip()
            disease = row[1].strip()

            disease_map[patient_id] = disease

    return disease_map


def get_patient_id(filename):
    return filename.split("_")[0]