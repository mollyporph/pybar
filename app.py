import cv2
import numpy as np
import time
from pyzbar import pyzbar

def sharpen_image(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def pad_4point_boundingbox(box, padding):
    newbox = box.copy()
    x_values = newbox.take(0, 1)
    left = np.argpartition(x_values, 2)[:2]
    right = np.argpartition(x_values, -2)[-2:]

    topleft, botleft = left[np.argpartition(newbox[left].take(1, 1), 1)]
    topright, botright = right[np.argpartition(newbox[right].take(1, 1), 1)]
    X = 0
    Y = 1
    newbox[topleft][X] = newbox[topleft][X] - padding
    newbox[topleft][Y] = newbox[topleft][Y] - padding

    newbox[botleft][X] = newbox[botleft][X] - padding
    newbox[botleft][Y] = newbox[botleft][Y] + padding

    newbox[topright][X] = newbox[topright][X] + padding
    newbox[topright][Y] = newbox[topright][Y] - padding

    newbox[botright][X] = newbox[botright][X] + padding
    newbox[botright][Y] = newbox[botright][Y] + padding
    return newbox


def main():
    imagepath = 'img/barcode.png'
    start = time.time()
    im = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    im = sharpen_image(im)
    barcodes = pyzbar.decode(im)
    if barcodes:
        print("Found barcode without preprocessing")
        for barcode in barcodes:
            print(barcode.rect)
            print(barcode.data.decode('utf-8'))
            print(barcode.type)
        end = time.time()
        print("Result after {}s".format(end-start))
        return
    im_gray = im.copy()

    im_out = cv2.imread(imagepath)

    scale = 800.0 / im.shape[1]
    im = cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)))

    kernel = np.ones((1, 3), np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_BLACKHAT, kernel, anchor=(1, 0))

    thresh, im = cv2.threshold(im, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((1, 8), np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_DILATE, kernel, anchor=(2, 0), iterations=2)

    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=2)

    kernel = np.ones((10, 18), np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    unscale = 1.0 / scale
    if contours is not None:
        for index, contour in enumerate(contours):

            if cv2.contourArea(contour) <= 2000:
                continue

            rect = cv2.minAreaRect(contour)
            rect = \
                ((int(rect[0][0] * unscale), int(rect[0][1] * unscale)),
                 (int(rect[1][0] * unscale), int(rect[1][1] * unscale)),
                 rect[2])

            box = np.int0(cv2.boxPoints(rect))
            box = pad_4point_boundingbox(box, 15)
            x1, x2, x3, x4 = box.take(0, 1)
            y1, y2, y3, y4 = box.take(1, 1)
            top_left_x = min([x1, x2, x3, x4])
            top_left_y = min([y1, y2, y3, y4])
            bot_right_x = max([x1, x2, x3, x4])
            bot_right_y = max([y1, y2, y3, y4])
            mask = np.zeros(im_gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [box], -1, (255, 255, 255), cv2.FILLED)
            crop = im_gray.copy()
            crop[True] = 0
            cv2.copyTo(im_gray, mask, crop)
            cv2.drawContours(im_gray, [box], 0, (0, 255, 0), thickness=2)

            crop = crop[top_left_y-2:bot_right_y+2, top_left_x-5:bot_right_x+5]
            angle = rect[2]
            rows, cols = crop.shape[0], crop.shape[1]
            angle = angle+90 if angle < -45 else angle
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            img_rot = cv2.warpAffine(crop, M, (cols, rows))
            sharpen = sharpen_image(img_rot)

            barcodes = pyzbar.decode(sharpen)
            if barcodes:
                print("Found barcode with preprocessing")
                for barcode in barcodes:
                    print(barcode.rect)
                    print(barcode.data.decode('utf-8'))
                    print(barcode.type)
                end = time.time()
                print("Result after {}s".format(end - start))
            cv2.drawContours(im_out, [box], 0, (0, 255, 0), thickness=2)

    cv2.imwrite('img/out.png', im_out)


if __name__ == "__main__":
    main()

