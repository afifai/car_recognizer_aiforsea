
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--log", required=True,
	help="Path to log training file")
args = vars(ap.parse_args())

f = open(args['log'])
content = f.readline()
train = []
train_k = []
val = []
val_k = []
train_loss = []
val_loss = []

while content != "":
	if 'Train-accuracy' in content:
		train.append(eval(content.split('=')[1]))
	elif 'Train-top_k_accuracy' in content:
		train_k.append(eval(content.split('=')[1]))
	elif 'Validation-accuracy' in content:
		val.append(eval(content.split('=')[1]))
	elif 'Validation-top_k_accuracy' in content:
		val_k.append(eval(content.split('=')[1]))
	elif 'Train-cross-entropy' in content:
		train_loss.append(eval(content.split('=')[1]))
	elif 'Validation-cross-entropy' in content:
		val_loss.append(eval(content.split('=')[1]))
	content = f.readline()

plt.style.use("ggplot")
plt.figure()
# plt.plot(np.arange(0, len(train)), train, label="train_acc")
# plt.plot(np.arange(0, len(train)), train_k, label="train_top_5_acc")
# plt.plot(np.arange(0, len(train)),val, label="val_acc")
# plt.plot(np.arange(0, len(train)), val_k, label="val_top_5_acc")
plt.plot(np.arange(0, len(train)), train_loss, label="train_loss")
plt.plot(np.arange(0, len(train)), val_loss, label="val_loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.show()