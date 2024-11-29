import numpy as np
import cv2
import os
import time

dirname = os.path.dirname(__file__)

def gaussianFilter(img, k, s):
    w, h, c = img.shape
    size = k // 2
    # パディング
    _img = np.zeros((w + 2 * size, h + 2 * size, c), dtype=np.float64)
    _img[size:size + w, size:size + h] = img.copy().astype(np.float64)
    for x in range(h):
        _img[0:size, size + x] = _img[size, size + x]
        _img[size + w:w + size * 2, size + x] = _img[size + w - 1, size + x]
    for y in range(w):
        _img[size + y, 0:size] = _img[size + y, size]
        _img[size + y, size + h:h + size * 2] = _img[size + y, size + h - 1]
    dst = _img.copy().astype(float)

    # フィルタ作成
    ker = np.zeros((k, k), dtype=np.float64)
    for x in range(-1 * size, k - size):
        for y in range(-1 * size, k - size):
            ker[x + size, y + size] = (1 / (2 * np.pi * (s ** 2))) * np.exp(-1 * (x ** 2 + y ** 2) / (2 * (s ** 2)))
    ker /= ker.sum()

    # フィルタリング処理
    dst = cv2.filter2D(dst, -1, ker)
    dst = dst[size:size + w, size:size + h].astype(np.uint8)
    return dst


def sub(before, after):
    before = before.copy().astype(np.float64)
    after = after.copy().astype(np.float64)
    dst = cv2.subtract(before, after) + 128
    return dst


def add(high, low, hrate, lrate):
    high = high.copy().astype(np.float64)
    low = low.copy().astype(np.float64)
    dst = cv2.add(high * hrate, low * lrate)
    return dst


def testadd():
    print("画像の画素値加算")
    name1 = input("画像名１を入力")
    name2 = input("画像名２を入力")
    name3 = input("合成後画像名を入力-半角英数のみ")
    start = time.time()
    img1 = cv2.imread(path + name1 + '.png')
    img2 = cv2.imread(path + name2 + '.png')
    dst = add(img1, img2, 0.5, 0.5)
    cv2.imwrite(path2 + name3 + '.png', dst)
    end = time.time()
    time_diff = end - start
    print(time_diff)
    print("処理終了")


def testgaus():
    print("画像の平滑化処理")
    name1 = input("画像名１を入力")
    name3 = input("処理後画像名を入力-半角英数のみ")
    no1 = int(input("フィルタサイズを入力"))
    no2 = int(input("σを入力"))
    start = time.time()
    img1 = cv2.imread(path + name1 + '.png')
    dst = gaussianFilter(img1, no1, no2)
    cv2.imwrite(path2 + name3 + '.png', dst)
    end = time.time()
    time_diff = end - start
    print(time_diff)
    print("処理終了")


def testsub():
    print("画像の画素値減算")
    name1 = input("減算元画像名１を入力")
    name2 = input("減算する画像名２を入力")
    name3 = input("減算後画像名を入力-半角英数のみ")
    start = time.time()
    img1 = cv2.imread(path + name1 + '.png')
    img2 = cv2.imread(path + name2 + '.png')
    dst = sub(img1, img2)
    cv2.imwrite(path2 + name3 + '.png', dst)
    end = time.time()
    time_diff = end - start
    print(time_diff)
    print("処理終了")

def videocomb(foldername, name):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    Video = cv2.VideoWriter("./Video/" + name + ".mp4", fourcc, 10.0, (1920, 1080))

    for i in range(0, 79 + 1):
        low_image = cv2.imread('./resources/' + foldername + '/' + name + '_%d.png' % i)
        if low_image is None:
            print("can't read")
            break
        Video.write(low_image)
    Video.release()
    print("written")

def videocut(name, no):
    video = cv2.VideoCapture("./Video/" + name + ".mp4")
    frameAll = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    for i in range(0, frameAll, 3):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, image = video.read()  # [ret]はread()の処理結果、[image]は処理画像が格納される
        if ret:
            cv2.imwrite('./resources/' + no + '/' + name + '_%d.png' % idx, image)
            idx += 1
        else:
            break


def roop_gaus(name, type, size, sig):
    if type == 0:
        for i in range(0, 79 + 1):
            img = cv2.imread("./resources/F_image/" + name + "_%d.png" % i)
            img2 = gaussianFilter(img, size, sig)
            cv2.imwrite("./resources/F_image_filtered/" + name + "_%d.png" % i, img2)
    if type == 1:
        for i in range(0, 79 + 1):
            img = cv2.imread("./resources/N_image/" + name + "_%d.png" % i)
            img2 = gaussianFilter(img, size, sig)
            cv2.imwrite("./resources/N_image_filtered/" + name + "_%d.png" % i, img2)


def makeVideo_From_VandV():
    start = time.time()
    lowname = input("低周波動画名を入力")
    highname = input("高周波動画名を入力")
    endname = input("完成時動画名を入力")
    videocut(lowname, "F_image")
    videocut(highname, "N_image")

    roop_gaus(lowname, 0, 240, 40)
    roop_gaus(highname, 1, 60, 10)

    for i in range(0, 79 + 1):
        img = cv2.imread("./resources/N_image_filtered/" + highname + "_%d.png" % i)
        img2 = cv2.imread("./resources/N_image/" + highname + "_%d.png" % i)
        img2 = sub(img2, img)
        cv2.imwrite("./resources/N_image_high/" + highname + "_%d.png" % i, img2)

    for i in range(0, 79 + 1):
        img = cv2.imread("./resources/F_image_filtered/" + lowname + "_%d.png" % i)
        img2 = cv2.imread("./resources/N_image_high/" + highname + "_%d.png" % i)
        img2 = add(img, img2, 0.5, 0.5)
        mask = cv2.imread("./resources/mask2/mask_%d.png" % i)
        cv2.imwrite("./resources/Hybrid_image/" + endname + "_%d.png" % i, img2)

    videocomb("Hybrid_image", endname)
    end = time.time()
    time_diff = end - start
    print(time_diff)


def makeHighVideo():
    start = time.time()
    highname = input("高周波動画名を入力")
    endname = input("完成時動画名を入力")
    videocut(highname, "N_image")
    roop_gaus(highname, 1, 60, 10)
    for i in range(0, 79 + 1):
        img = cv2.imread("./resources/N_image_filtered/" + highname + "_%d.png" % i)
        img2 = cv2.imread("./resources/N_image/" + highname + "_%d.png" % i)
        img2 = sub(img2, img)
        cv2.imwrite("./resources/N_image_high/" + endname + "_%d.png" % i, img2)
    videocomb("N_image_high", endname)
    end = time.time()
    time_diff = end - start
    print(time_diff)

def makeLowVideo():
    start = time.time()
    lowname = input("低周波動画名を入力")
    videocut(lowname,"F_image")
    roop_gaus(lowname, 0, 240, 40)
    videocomb("F_image_filtered",lowname)
    end = time.time()
    time_diff = end - start
    print(time_diff)


path = ('./test-image/')
path2 = ('./after-image/')

def answer():
    what=int(input("作りたいものがハイブリッド映像なら0, 高周波映像なら1, 低周波映像なら2を入力"))
    if(what==0):
        makeVideo_From_VandV()
    elif(what==1):
        makeHighVideo()
    elif(what==2):
        makeLowVideo()
    else:
        print("入力失敗")

answer()