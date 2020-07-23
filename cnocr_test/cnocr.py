# -*- coding: utf-8 -*-


from cnocr import CnOcr  # pip install cnocr  # 中文OCR
import cv2


cn_ocr = CnOcr(root='cnocr_model')

img = cv2.imread('data/tt14.png')
ocr_res = cn_ocr.ocr_for_single_line(img)
print('ocr result: %s' % ''.join(ocr_res))
