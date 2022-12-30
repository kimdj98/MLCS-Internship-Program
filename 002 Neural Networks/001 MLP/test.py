import train
import pdb

import matplotlib.pyplot as plt

history1 = train.history1
history2 = train.history2

score1 = train.score1
score2 = train.score2

print('')
print('Train accuracy for model1:', history1.history['accuracy'][-1])
print('Test loss for model1:', score1[0])
print('Test accuracy for model1:', score1[1])
# accuracy: 0.9775238037109375

print('')
print('Train accuracy for model2:', history2.history['accuracy'][-1])
print('Test loss for model2:', score2[0])
print('Test accuracy for model2:', score2[1])
# accuracy: 0.9832381010055542

# plot model1 accuracy plot
plt.plot(history1.history["accuracy"], label="accuracy")
plt.plot(history1.history["val_accuracy"], label="val_accuracy")
plt.title("model1_accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend()
plt.show()

# plot model2 accuracy plot
plt.plot(history2.history["accuracy"], label="accuracy")
plt.plot(history2.history["val_accuracy"], label="val_accuracy")
plt.title("model2_accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend()
plt.show()
pdb.set_trace()