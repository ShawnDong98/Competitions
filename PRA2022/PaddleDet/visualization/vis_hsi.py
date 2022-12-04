from scipy import io as sio
import wx

app = wx.App()

import spectral
from spectral import *
import cv2


spectral.settings.WX_GL_DEPTH_SIZE = 16

HSI = sio.loadmat('./datasets/HSI/hsi/newrawfile20220220144358.mat')['dest']
view_cube(HSI, bands=[29, 19, 9])

cv2.waitKey(0)