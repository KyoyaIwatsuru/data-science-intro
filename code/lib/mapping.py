import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plotScanPath(
        X, Y, durations, figsize=(30, 15),
        bg_image="", save_path="", halfPage=False):
  plt.figure(figsize=figsize)
  if bg_image != "":
    img = mpimg.imread(bg_image)
    plt.imshow(img)
    if halfPage:
      plt.xlim(150, 1000)
    else:
      plt.xlim(0, len(img[0]))
    plt.ylim(len(img), 0)
  scale = float(figsize[0]) / 40.0

  plt.plot(X[0:2], Y[0:2], "-", c="blue", linewidth=scale, zorder=1, alpha=0.8)
  plt.plot(X[2:4], Y[2:4], "-", c="blue", linewidth=scale, zorder=1, alpha=0.8)
  plt.plot(X[4:6], Y[4:6], "-", c="blue", linewidth=scale, zorder=1, alpha=0.8)
  plt.plot(X[6:8], Y[6:8], "-", c="blue", linewidth=scale, zorder=1, alpha=0.8)
  plt.plot(X[8:10], Y[8:10], "-", c="blue", linewidth=scale, zorder=1, alpha=0.8)
  # plt.scatter(X, Y, durations*scale, c="b", zorder=2, alpha=0.3)
  # plt.scatter(X[0], Y[0], durations[0]*scale, c="g", zorder=2, alpha=0.6)
  # plt.scatter(X[-1], Y[-1], durations[-1]*scale, c="r", zorder=2, alpha=0.6)
  plt.show()

target_dir = "./data/input/img/main2/"
image_path = target_dir+"008_back.png"
X = [100.0, 1800.0, 100.0, 1800.0, 100.0, 1800.0, 100.0, 1800.0, 100.0, 1800.0]
Y = [260.0, 260.0, 435.0, 435.0, 575.0, 575.0, 715.0, 715.0, 855.0, 855.0]

plotScanPath(X, Y, 50.0, bg_image=image_path)