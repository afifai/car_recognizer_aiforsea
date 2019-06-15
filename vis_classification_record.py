# USAGE
# python vis_classification.py --checkpoints checkpoints --prefix vggnet --epoch 55

# import the necessary packages
import cv2
from config import car_config as config
from utility.preprocessing import ImageToArrayPreprocessor
from utility.preprocessing import AspectAwarePreprocessor
from utility.preprocessing import MeanPreprocessor
from utility.utils import process_image as get_color
import numpy as np
import mxnet as mx
import argparse
import pickle
import imutils
import os
from PIL import Image
f = open("fail_log.txt", "w")

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
	help="path to the checkpoint directory")
ap.add_argument("-p", "--prefix", required=True,
	help="name of model prefix")
ap.add_argument("-e", "--epoch", type=int, required=True,
	help="epoch # to load")
ap.add_argument("-s", "--sample-size", type=int, default=50,
	help="epoch # to load")
args = vars(ap.parse_args())

le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())
rows = open(config.TEST_MX_LIST).read().strip().split("\n")
#rows = np.random.choice(rows, size=args["sample_size"])

print("[INFO] loading pre-trained model...")
checkpointsPath = os.path.sep.join([args["checkpoints"],
	args["prefix"]])
model = mx.model.FeedForward.load(checkpointsPath,
	args["epoch"])

model = mx.model.FeedForward(
	ctx=[mx.gpu(0)],
	symbol=model.symbol,
	arg_params=model.arg_params,
	aux_params=model.aux_params)

sp = AspectAwarePreprocessor(width=224, height=224)
mp = MeanPreprocessor(config.R_MEAN, config.G_MEAN, config.B_MEAN)
iap = ImageToArrayPreprocessor(dataFormat="channels_first")

for row in rows:
	(target, imagePath) = row.split("\t")[1:]
	target = int(target)

	image = cv2.imread(imagePath)
	orig = image.copy()
	orig = imutils.resize(orig, width=min(500, orig.shape[1]))
	image = iap.preprocess(mp.preprocess(sp.preprocess(image)))
	image = np.expand_dims(image, axis=0)

	preds = model.predict(image)[0]
	idxs = np.argsort(preds)[::-1][:5]

	print("[INFO] actual={}".format(le.inverse_transform([target])[0]))

	label = le.inverse_transform([idxs[0]])[0]
	label = label.replace(":", " ")
	label = "{}: {:.2f}%".format(label, preds[idxs[0]] * 100)
	try:
		img_pil = Image.open(imagePath)
		color = "color : " + get_color(img_pil)
		cv2.putText(orig, color, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
		0.6, (255, 0, 200), 2)
	except Exception as e:
		f.write(row.split("\t")[1:][1].split("/")[-1])
		f.write("\n")
		f.write(str(e))
		f.write("\n")

	cv2.putText(orig, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		0.6, (0, 255, 0), 2)
	

	for (i, prob) in zip(idxs, preds):
		print("\t[INFO] predicted={}, probability={:.2f}%".format(
			le.inverse_transform([i])[0], preds[i] * 100))

	print("[INFO] predicted color={}".format(color.split(": ")[1]))
#	cv2.imshow("Image", orig)
#	cv2.waitKey(0)
	cv2.imwrite("res/Test"+row.split("\t")[1:][1].split("/")[-1], orig)
f.close()