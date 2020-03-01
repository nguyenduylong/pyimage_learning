from keras.models import load_model
import coremltools
import argparse
import pickle
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")

args = vars(ap.parse_args())

model = load_model(args['model'])

coreml_model = coremltools.converters.keras.convert(model,
	input_names="image",
	image_input_names="image",
	image_scale=1/255.0,
	class_labels= np.arange(0, 12, 1).tolist(),
	is_bgr=True)

output = args["model"].rsplit(".", 1)[0] + ".mlmodel"

coreml_model.save(output)

