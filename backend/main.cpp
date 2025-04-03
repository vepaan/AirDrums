#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    VideoCapture cap;
    int max_resolution = 0;
    int best_cam_idx = -1;

    //we find the best available camera and use that
    for (int i = 0; i < 10; i++) {
        VideoCapture temp(i);
        if (temp.isOpened()) {
            int width = temp.get(CAP_PROP_FRAME_WIDTH);
            int height = temp.get(CAP_PROP_FRAME_HEIGHT);
            int res = width * height;

            cout << "Cam " << i << "detected with res: " << width << "x" << height << endl;

            if (res > max_resolution) {
                max_resolution = res;
                best_cam_idx = i;
            }

            temp.release();
        }
    }

    if (best_cam_idx == -1) {
        cerr << "No camera detected!" << endl;
        return -1;
    }

    //opening best cam
    cap.open(best_cam_idx);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera" << endl;
        return -1;
    }

    cout << "Using camera" << best_cam_idx << endl;

    Mat frame;

    while (true) {
        cap >> frame;

        if (frame.empty()) {
            cerr << "Cant receive data" << endl;
            break;
        }

        imshow("AirDrums", frame);
        if (waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
