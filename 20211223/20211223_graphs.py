import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
loss_lst = [0.8660, 0.2353, 0.1712, 0.1432, 0.1243, 0.1116, 0.1006, 0.0921, 0.0871, 0.0705]

ax.plot(range(10), loss_lst)
ax.set_xlabel('Epoches', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.grid(color='grey', linestyle='dashed', linewidth=1, alpha=0.3)
fig.show()