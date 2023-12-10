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
