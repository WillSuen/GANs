import numpy as np
import cv2


def fill_buf(buf, i, img, shape):
    n = buf.shape[0] / shape[1]
    m = buf.shape[1] / shape[0]

    sx = (i % m) * shape[0]
    sy = (i / m) * shape[1]
    buf[sy:sy + shape[1], sx:sx + shape[0], :] = img


def visual(title, X):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0]))
    n = n.astype(int)
    buff = np.zeros((n*X.shape[1], n*X.shape[2], X.shape[3]), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    buff = cv2.cvtColor(buff, cv2.COLOR_BGR2RGB)
    cv2.imwrite(title, buff)
    cv2.waitKey(1)
