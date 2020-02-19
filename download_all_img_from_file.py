from imutils import paths
import argparse
import requests
import cv2
import os
import json

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='path to json input file')
ap.add_argument('-o', '--output', required=True, help='path to output directory of images')
args = vars(ap.parse_args())

data_rows = open(args['input']).read().strip().splitlines()
total = 0;
folder_names= []
for data_row in data_rows:
    parsed_json = json.loads(data_row);
    content = parsed_json['content'];
    labels = parsed_json['annotation']['labels'];
    if not labels:
        continue
    output_folder = os.path.join(args['output'], labels[0])
    if labels[0] not in folder_names:
        os.mkdir(output_folder)
        folder_names.append(labels[0])
    try:
        r = requests.get(content, timeout=60)
        p = os.path.join(output_folder, '{}.jpg').format(str(total).zfill(8))
        f = open(p, 'wb')
        f.write(r.content)
        f.close()
        print("[INFO] downloaded: {}".format(p))
        total += 1
    except:
        print("[INFO] error downloading {}...skipping".format(p))

        