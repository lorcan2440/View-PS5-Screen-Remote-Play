import numpy as np
from ctypes import windll
import win32gui, win32ui
import cv2 as cv

WIN_HANDLES = None
PW_CLIENTONLY = 0x03  # https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-printwindow


def capture_win_alt(window_name: str):

    global WIN_HANDLES

    if WIN_HANDLES is None or cv.waitKey(1) == ord('r'):
        # press 'r' to refresh after changing the window size
        windll.user32.SetProcessDPIAware()
        hwnd = win32gui.FindWindow(None, window_name)
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        w = right - left
        h = bottom - top
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
        WIN_HANDLES = (hwnd, hwnd_dc, mfc_dc, save_dc, bitmap)

    (hwnd, hwnd_dc, mfc_dc, save_dc, bitmap) = WIN_HANDLES
    save_dc.SelectObject(bitmap)

    # If Special K is running, this number is 3. If not, 1
    res = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_CLIENTONLY)  # <winuser.h>
    bmpinfo = bitmap.GetInfo()
    bmpstr = bitmap.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
    img = np.ascontiguousarray(img)[..., :-1]  # make image C_CONTIGUOUS and drop alpha channel
    
    if res != 1:
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        raise RuntimeError(f"Unable to acquire screenshot! Result: {res}")

    return img


cv.namedWindow('Computer Vision', cv.WINDOW_NORMAL)
while cv.waitKey(1) != ord('q'):
    # press 'q' to quit
    img = capture_win_alt('PS Remote Play')
    cv.imshow('Computer Vision', img)
