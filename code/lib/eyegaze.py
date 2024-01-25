import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import multivariate_normal
import matplotlib.cm as cm


def isMinimumFixation(X, Y, mfx):
  if max([max(X) - min(X), max(Y) - min(Y)]) < mfx:
    return True
  return False


def detectFixations(
  times, X, Y, P,
  min_concat_gaze_count=9,
  min_fixation_size=50,
  max_fixation_size=80):

  fixations = []
  i = 0
  j = 0
  while max([i, j]) < len(times)-min_concat_gaze_count:
    X_ = list(X[i:i+min_concat_gaze_count])
    Y_ = list(Y[i:i+min_concat_gaze_count])
    P_ = list(P[i:i+min_concat_gaze_count])
    if isMinimumFixation(X_, Y_, min_fixation_size):
      j = i + min_concat_gaze_count
      c = 0
      begin = times[i]
      end = times[j - 1]
      while(c < min_concat_gaze_count and j < len(times)):
        X_.append(X[j])
        Y_.append(Y[j])
        P_.append(P[j])
        if max([max(X_) - min(X_), max(Y_) - min(Y_)]) > max_fixation_size:
          # X[j]Y[j] is out of max_fixation_size
          if c == 0:
            # X[j]Y[j] will be next minimum fixation
            i = j
          X_.pop()
          Y_.pop()
          P_.pop()
          c += 1
        else:
          c = 0
          end = times[j]
        j += 1
      i = i - 1
      j = j - 1
      fixations.append([begin, np.mean(X_), np.mean(Y_), end - begin, np.mean(P_)])
    i += 1

  lengths = [0.0]
  angles = [0.0]
  durations = [1.0]
  for i in range(1, len(fixations)):
    delta_x = fixations[i][1] - fixations[i-1][1]
    delta_y = fixations[i][2] - fixations[i-1][2]
    delta_t = fixations[i][0] - (fixations[i-1][0] + fixations[i-1][3])
    if delta_t == 0.0:
      delta_t = 1.0

    lengths.append(math.sqrt(delta_x * delta_x + delta_y * delta_y))
    angles.append(math.degrees(math.atan2(delta_y, delta_x)))
    durations.append(delta_t)

  saccades = np.vstack((lengths, angles, np.array(lengths) / np.array(durations))).T
  results = np.hstack((fixations, saccades))

  return np.array(results)


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

  plt.plot(X, Y, "-", c="blue", linewidth=scale, zorder=1, alpha=0.8)
  plt.scatter(X, Y, durations*scale, c="b", zorder=2, alpha=0.3)
  plt.scatter(X[0], Y[0], durations[0]*scale, c="g", zorder=2, alpha=0.6)
  plt.scatter(X[-1], Y[-1], durations[-1]*scale, c="r", zorder=2, alpha=0.6)

  if save_path != "":
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def plotHeatmap(
    X, Y, durations, figsize=(30, 15),
    bg_image="", save_path="", data_save_path=""):
  plt.figure(figsize=figsize)
  if bg_image != "":
    img = mpimg.imread(bg_image)
    plt.imshow(img)
    plt.xlim(0, len(img[0]))
    plt.ylim(len(img), 0)

  gx, gy = np.meshgrid(np.arange(0, len(img[0])), np.arange(0, len(img)))
  values = np.zeros((len(img), len(img[0])))
  for i in range(len(X)):
    # 設定された分散値を50に固定
    covariance = np.eye(2) * 50
    # 2次元の多変量正規分布を作成
    mv_normal = multivariate_normal(mean=[X[i], Y[i]], cov=covariance)
    # 密度関数を計算し、durationで重み付け
    density = mv_normal.pdf(np.dstack((gx, gy)))
    values += density * durations[i] / 2.0
  values = values/np.max(values)

  masked = np.ma.masked_where(values < 0.05, values)
  cmap = cm.jet
  cmap.set_bad('white', 1.)
  plt.imshow(masked, alpha=0.4, cmap=cmap)

  if save_path != "":
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()

  if data_save_path != "":
    np.savetxt(data_save_path, values, delimiter=",", fmt="%f")
