# Livestream PS5 Gameplay into OpenCV running on PC

A lot of existing scripts to read graphics from windows into Python do not work for windows that are both hardware-accelerated (such as PS Remote Play) and obscured by other windows (always useful when wanting to run in the background). Here is one that does! Scripts provided in Python and C++ (user's choice).

## Requirements:

- PS Remote Play Software: https://remoteplay.dl.playstation.net/remoteplay/lang/gb/index.html
- PC running Windows with Python installed

## Steps

I followed the steps in the first 3 minutes only of [this](https://www.youtube.com/watch?v=cNBs8Wgelf0) video, except for the OBS/Streamlabs part.

The steps are:

1. Create a new PSN account ("the second account"). This will be used to host the Remote Play session to decrease lag. You will still be able to use your main account to actually play the game!
2. While logged into the second account, go to Settings > Users and Accounts > Other > Enable Console Sharing and Offline Play.
3. Switch users (don't log out) back to your main account.
4. Install [PS Remote Play](https://remoteplay.dl.playstation.net/remoteplay/lang/gb/index.html) on your PC using your platform. Only tested on Windows.
5. Run the PS Remote Play software on your PC and sign into it using the second account.
6. Connect to your PS5 by following the instructions in the app. This can take a few minutes.
7. Press a button on your controller to bring up the "Who's playing?" box - select your main account.
8. To remove the "Remote Play Connected" banner at the top of the screen, on your PS5 main account, go to Settings > System > Language and choose any other language, then change it back to your own language. This seems to be a workaround for a bug rather than an actual feature so this may not be necessary in future.
9. You should now see the live video feed on your PC in PS Remote Play, and be able to play using the controller on your main account. Keep the Remote Play window active (*not minimised* - just put it behind whatever other windows you need to look at).
10. In python, install the libraries NumPy, PyWin32 and OpenCV. The pip command is:
    `pip install numpy opencv-python pywin32`
11. Run the python code in `record_window.py`. If all goes well you should see your gameplay live in an OpenCV window. You can of course use this code as a module in your larger project.

The latency is low with PS Remote Play. The resolution depends on your internet connection but is usually pretty good. The frame rate is usually not as high as the source but should be 30+ FPS. It's good enough that you can play even fast-paced games by watching the stream instead of your TV.

You can get even faster processing by using C++ instead of Python. I have provided an equivalent version of the python script here in C++. Installing opencv in C++ requires building the library from scratch however which is not as beginner friendly as python.

Great for computer vision applications (object detection etc).

## Code (Python) `record_window.py`

```python

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
    - `window_title` (str): The title of the window to capture. Hovering
    over the window in the taskbar will usually (not always) show the correct name to use.
    
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


if __name__ == '__main__':
    cv.namedWindow('Computer Vision', cv.WINDOW_NORMAL)
    while cv.waitKey(1) != ord('q'):
        # press 'q' to quit
        img = capture_win_alt('PS Remote Play')
        cv.imshow('Computer Vision', img)
```

## Code (C++) `record_window.cpp`

```cpp

#include <iostream>
#include <Windows.h>
#include <opencv2/opencv.hpp>

cv::Mat captureWindow(const std::string& windowName) {
    HWND hwnd = FindWindowA(NULL, windowName.c_str());

    RECT rect;
    GetClientRect(hwnd, &rect);
    int width = rect.right - rect.left;
    int height = rect.bottom - rect.top;

    HDC hdcScreen = GetDC(hwnd);
    HDC hdcMem = CreateCompatibleDC(hdcScreen);
    HBITMAP hBitmap = CreateCompatibleBitmap(hdcScreen, width, height);
    HGDIOBJ hOld = SelectObject(hdcMem, hBitmap);

    // If Special K is running, this number is 3. If not, 1
    int result = PrintWindow(hwnd, hdcMem, 3);

    BITMAPINFOHEADER bi;
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height;  // Negative height to ensure correct orientation
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0;
    bi.biXPelsPerMeter = 0;
    bi.biYPelsPerMeter = 0;
    bi.biClrUsed = 0;
    bi.biClrImportant = 0;

    BYTE* pBuffer = new BYTE[width * height * 4];
    GetDIBits(hdcMem, hBitmap, 0, height, pBuffer, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

    cv::Mat img(height, width, CV_8UC4, pBuffer);

    if (!result) {
        DeleteObject(hBitmap);
        DeleteDC(hdcMem);
        ReleaseDC(hwnd, hdcScreen);
        delete[] pBuffer;
        throw std::runtime_error("Unable to acquire screenshot! Result: " + std::to_string(result));
    }

    // Drop alpha channel
    cv::Mat imgBGR;
    cv::cvtColor(img, imgBGR, cv::COLOR_BGRA2BGR);

    DeleteObject(hBitmap);
    DeleteDC(hdcMem);
    ReleaseDC(hwnd, hdcScreen);
    delete[] pBuffer;

    return imgBGR;
}

int main() {
    const std::string WINDOW_NAME = "PS Remote Play";

    while (cv::waitKey(1) != 'q') {
        cv::Mat screenshot = captureWindow(WINDOW_NAME);
        cv::imshow("Computer Vision", screenshot);
    }

    return 0;
}
```
