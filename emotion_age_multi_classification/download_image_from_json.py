from imutils import paths
import argparse
import requests
import cv2
import os
import json

input = './dataset/compress_file/Frontal_Face.json'
output = './dataset/download_data/'

def getUsingLabelFromLists(labels):
    gender_label = ''
    emotion_label = ''
    for label in labels:
        if 'Emotion' in label:
            emotion_label = label.split('_',1)[-1]
        if 'Gender' in label:
            gender_label = label.split('_', 1)[-1]
    return gender_label, emotion_label

data_rows = open(input).read().strip().splitlines()
total = 0;
folder_names= []
for data_row in data_rows:
    parsed_json = json.loads(data_row)
    content = parsed_json['content']
    annotation_data = parsed_json['annotation']
    if  annotation_data is None or 'labels' not in annotation_data:
        continue
    labels = annotation_data['labels'];
    if len(labels) <= 1:
        continue
    gender_label, emotion_label = getUsingLabelFromLists(labels)
    if gender_label == '' or emotion_label == '':
        continue
    saved_folder_name = "{}_{}".format(emotion_label, gender_label)
    saved_folder_path = os.path.join(output, saved_folder_name)
    if saved_folder_name not in folder_names:
        os.mkdir(saved_folder_path)
        folder_names.append(saved_folder_name)
    try:
        r = requests.get(content, timeout=60)
        p = os.path.join(saved_folder_path, '{}.jpg').format(str(total).zfill(8))
        f = open(p, 'wb')
        f.write(r.content)
        f.close()
        print("[INFO] downloaded: {}".format(p))
        total += 1
    except:
        print("[INFO] error downloading {}...skipping".format(p))
