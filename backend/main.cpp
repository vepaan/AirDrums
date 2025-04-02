#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    Mat img = imread("test.png");
    if (img.empty()) {
        cerr << "Image not found!" << endl;
        return -1;
    }
    imshow("Test Image", img);
    waitKey(0);
    return 0;
}
