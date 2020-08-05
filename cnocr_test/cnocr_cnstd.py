# -*- coding: utf-8 -*-


from cnstd import CnStd  # pip install cnstd  # 文本检测
from cnocr import CnOcr  # pip install cnocr  # 中文OCR
import cv2  # pip install opencv-python


# cn_std = CnStd()
# cn_ocr = CnOcr()
cn_std = CnStd(root='cnstd_model')  # 删除 cnstd_model 目录下的模型，再运行，可重新下载
cn_ocr = CnOcr(root='cnocr_model')


# box_info_list = cn_std.detect('data/tt09.png')
box_info_list = cn_std.detect('data/tt14.png')

for box_info in box_info_list:
    cropped_img = box_info['cropped_img']  # 检测出的文本框

    # cv2.imshow('cropped_img', cropped_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ocr_res = cn_ocr.ocr_for_single_line(cropped_img)
    print('ocr result: %s' % ''.join(ocr_res))
