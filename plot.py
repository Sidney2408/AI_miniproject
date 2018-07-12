import pickle
import matplotlib.pyplot as plt

with open("stats.pkl","rb") as f:
    model_stats = pickle.load(f)

# Hardcoded, manually copied from log.
train_losses = [3.27053, 2.80648, 2.68175, 2.62595, 2.59937, 2.58393] 

losses = []
acc = []
for i in range(len(model_stats)):
    model = "model{}.pth".format(i)
    l,a = model_stats[model]
    losses.append(l)
    acc.append(a)

epochs = range(len(losses))
for i, tl, l, a in zip(epochs, train_losses, losses, acc):
    print("Model{}: tl:{:.5f}  vl:{:.5f}  va:{:.5f}".format(i,tl,l,a))


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig,ax = plt.subplots(figsize=(8,6))
ax.plot(epochs, train_losses, "-", color=colors[0], label="Train Loss")
ax.plot(epochs, losses, "-", color=colors[1], label="Validation Loss")

ax2 = ax.twinx()
ax2.plot(epochs, acc, "-", color=colors[2], label="Validation Accuracy")

ax.set_xticks(epochs)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax2.set_ylabel("Accuracy")
#y_bottom, y_top = ax2.get_ylim()
#y_top += (y_top-y_bottom)*0.1
#ax2.set_ylim((y_bottom, y_top))
ax2.set_ylim((0, 0.7))


handles, labels = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles.extend(handles2)
labels.extend(labels2)
ax.legend(handles, labels, loc=1)

fig.tight_layout()
plt.show()