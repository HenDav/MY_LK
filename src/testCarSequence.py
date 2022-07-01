import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanadeHenVicImprovements import LucasKanade as LKOurs, PyrLK
from LucasKanade import LucasKanade
import time

# write your script here, we recommend the above libraries for making your animation
frames = np.load('../data/carseq.npy')
rect = np.array([59.0, 116.0, 145.0, 151.0])
width = rect[3] - rect[1]
length = rect[2] - rect[0]
wh = np.array([length, width]).astype(float)
xy = rect[:2] + wh * 0.5
rectList = []
time_total = 0
seq_len = frames.shape[2]

for i in range(seq_len):
    if (i == 0):
        continue
    print("Processing frame %d" % i)
    a = rect.copy()
    rectList.append(a)

    start = time.time()
    It = frames[:,:,i-1]
    It1 = frames[:,:,i]
    print("xy: ", xy)
    p_orig = LucasKanade(It, It1, rect)
    print("p_orig: ", p_orig)
    p, err = PyrLK(It, It1, xy, wh, 1)
    print("p: ", p)
    print("err: ", err)
    xy += p
    rect = np.concatenate([xy - wh * 0.5, xy + wh * 0.5])
    # rect[:2] += p_orig
    # rect[:2] += p_orig
    # rect[2:4] += p_orig
    # xy = rect[:2] + wh * 0.5

    end = time.time()
    time_total += end - start

    if i % 2 == 0 or i == 1:
        plt.figure()
        plt.imshow(frames[:,:,i],cmap='gray')
        bbox = patches.Rectangle((int(rect[0]), int(rect[1])), length, width,
                                 fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(bbox)
        plt.title('frame %d'%i)
        plt.show()
np.save('carseqrects.npy',rectList)
print('Finished, the tracking frequency is %.4f' % (seq_len / time_total))
