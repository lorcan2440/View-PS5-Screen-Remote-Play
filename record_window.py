import numpy as np
from ctypes import windll
import win32gui, win32ui
import cv2 as cv

WIN_HANDLES = None
PW_CLIENTONLY = 0x03  # https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-printwindow


def capture_win_alt(window_title: str) -> np.ndarray:
    '''
    Read an image frame from a given named window. The window must not be
    minimised, but it can be obscured behind other windows. It can also
    be hardware accelerated.

    This function works well for capturing gameplay from PS Remote Play. Example:

    ```python
    if __name__ == '__main__':
        cv.namedWindow('Computer Vision', cv.WINDOW_NORMAL)
        while cv.waitKey(1) != ord('q'):
            # press 'q' to quit
            img = capture_win_alt('PS Remote Play')
            cv.imshow('Computer Vision', img)
    ```
    
    ### Arguments
    - `window_title` (str): The title of the window to capture. Use the function
    `get_all_window_names()` to get a list of all windows. Alternatively, hovering
    over the window in the taskbar will usually show the correct name to use.
    
    ### Returns
    - `np.ndarray`: a 3D BGR image array of shape (height, width, 3).

    ### Raises
    - `RuntimeError`: if something happens to prevent the image in the window being read.
    Common cause: minimising the window.
    '''    

    global WIN_HANDLES

    if WIN_HANDLES is None or cv.waitKey(1) == ord('r'):
        # press 'r' to refresh after changing the window size
        windll.user32.SetProcessDPIAware()
        hwnd = win32gui.FindWindow(None, window_title)
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


def get_all_window_names():
    import pyautogui
    windows = pyautogui.getAllWindows()
    for window in windows:
        print(window.title)

if __name__ == '__main__':
    get_all_window_names()
    cv.namedWindow('Computer Vision', cv.WINDOW_NORMAL)
    while cv.waitKey(1) != ord('q'):
        # press 'q' to quit
        img = capture_win_alt('PS Remote Play')
        cv.imshow('Computer Vision', img)
