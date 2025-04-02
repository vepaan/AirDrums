#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("test.png");
    if (img.empty()) {
        std::cerr << "Image not found!" << std::endl;
        return -1;
    }
    cv::imshow("Test Image", img);
    cv::waitKey(0);
    return 0;
}
