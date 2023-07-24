#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>

using namespace std;
using namespace cv;

void drawFaceCount(Mat& img, int count) {
    rectangle(img, Point(img.cols - 250, 0), Point(img.cols, 70), Scalar(0, 255, 0), FILLED);
    putText(img, to_string(count) + " Faces Found", Point(img.cols - 250, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2);
}

int main() {
        VideoCapture video(1);
    Mat img;

    const std::string modelConfiguration = "./deploy.prototxt";
    const std::string modelWeights = "./res10_300x300_ssd_iter_140000_fp16.caffemodel";

    cv::dnn::Net net = cv::dnn::readNetFromCaffe(modelConfiguration, modelWeights);

    if (net.empty()) {
        cout << "Error: Failed to load the DNN model." << endl;
        return -1;
    }

    if (!video.isOpened()) {
        cout << "Error: Failed to open the camera." << endl;
        return -1;
    }

    vector<Rect> prev_faces; // Previous detected faces

    while (true) {
        video.read(img);

        if (img.empty()) {
            cout << "Error: Failed to capture frame." << endl;
            break;
        }

        cv::Mat inputBlob = cv::dnn::blobFromImage(img, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);

        net.setInput(inputBlob, "data");
        cv::Mat detection = net.forward("detection_out");

        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        vector<Rect> faces;
        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);

            if (confidence > 0.5) { // Confidence threshold (adjust as needed)
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);

                faces.push_back(Rect(Point(x1, y1), Point(x2, y2)));
            }
        }

        // Calculate the average position of the detected faces
        if (!prev_faces.empty()) {
            for (int i = 0; i < faces.size(); ++i) {
                faces[i].x = (faces[i].x + prev_faces[i].x) / 2;
                faces[i].y = (faces[i].y + prev_faces[i].y) / 2;
            }
        }

        prev_faces = faces; // Store the current faces for the next iteration

        // Draw the detected faces
        for (const Rect& face : faces) {
            rectangle(img, face.tl(), face.br(), Scalar(0, 255, 0), 3);
        }

        drawFaceCount(img, faces.size());

        // Calculate and display FPS
        static auto start = chrono::high_resolution_clock::now();
        auto end = chrono::high_resolution_clock::now();
        double fps = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        fps = 1000.0 / fps; // Convert to frames per second
        start = end;

        stringstream ss;
        ss << "FPS: " << fixed << setprecision(1) << fps;
        putText(img, ss.str(), Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        imshow("Frame", img);

        char key = waitKey(30); // Increased delay to stabilize video display
        if (key == 'q') {
            break;
        }
    }

    return 0;
}
