// header files

#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>
#include <utility>
#include <stdio.h>


using namespace std;
using namespace cv;

void detectAndDisplay(Mat frame, Mat framegt);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

int main(){

    // 5 images given to detect faces

    for (int i = 0; i < 5; i++){

        int j[5] = {4, 5, 13, 14, 15};

        string input = "dart";
        input = input + to_string(j[i]) + ".jpg";

        string inputgt = "dartgtf";
        inputgt = inputgt + to_string(j[i]) + ".png";

        // 1. Read Input Image and Ground truth Image
        Mat frame = imread(input, CV_LOAD_IMAGE_COLOR);
        Mat framegt = imread(inputgt, CV_LOAD_IMAGE_COLOR);


        // 2. Load the Strong Classifier in a structure called `Cascade'
        if(!cascade.load(cascade_name)){ printf("--(!)Error loading\n"); return -1; };

        // 3. Detect faces and Display Result
        detectAndDisplay(frame, framegt);

        string output = "FaceDetectOutputtest";
        output = output + to_string(j[i]) + ".jpg";
        imwrite(output, frame);
    }
    return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame, Mat framegt){

    std::vector<Rect> faces;
    Mat gray;

    // 1. Prepare Image by turning it into Grayscale and normalising lighting
    cvtColor(frame, gray, CV_BGR2GRAY);
    equalizeHist(gray, gray);

    // 2. Perform Viola-Jones Object Detection
    cascade.detectMultiScale(gray, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50,50), Size(500,500));

    int size = faces.size();

    // 3. Print number of Faces found by the detector
    std::cout << size << std::endl;

    //Run over all detections
    for (int d = 0; d < size; d++){

        int x_top = faces[d].x;
        int y_top = faces[d].y;
        int x_bot = faces[d].x + faces[d].width;
        int y_bot = faces[d].y + faces[d].height;

        rectangle(frame, Point(x_top, y_top), Point(x_bot, y_bot), Scalar(255, 255, 0), 2);

       //Set tolerance value based on detection frame size
       int x_tol = floor(0.3*faces[d].width);
       int y_tol = floor(0.3*faces[d].width);

       for (int k1 = x_top-x_tol; k1 <= x_top+x_tol; k1++){
          for (int l1 = y_top-y_tol; l1 <= y_top+y_tol; l1++){
             //Determine if pixel is top left corner
             if ((framegt.at<Vec3b>(l1,k1)[0]   == 0) &&
                 (framegt.at<Vec3b>(l1,k1)[1]   == 255) &&
                 (framegt.at<Vec3b>(l1,k1)[2]   == 0) &&
                 (framegt.at<Vec3b>(l1+1,k1)[0] == 0) &&
                 (framegt.at<Vec3b>(l1+1,k1)[1] == 255) &&
                 (framegt.at<Vec3b>(l1+1,k1)[2] == 0) &&
                 (framegt.at<Vec3b>(l1-1,k1)[0] != 0) &&
                 (framegt.at<Vec3b>(l1-1,k1)[1] != 255) &&
                 (framegt.at<Vec3b>(l1-1,k1)[2] != 0) &&
                 (framegt.at<Vec3b>(l1,k1+1)[0] == 0) &&
                 (framegt.at<Vec3b>(l1,k1+1)[1] == 255) &&
                 (framegt.at<Vec3b>(l1,k1+1)[2] == 0) &&
                 (framegt.at<Vec3b>(l1,k1-1)[0] != 0) &&
                 (framegt.at<Vec3b>(l1,k1-1)[1] != 255) &&
                 (framegt.at<Vec3b>(l1,k1-1)[2] != 0)){
                int k2 = k1;
                int l2 = l1;

                //Run along the top of the rectangle
                do{
                   k2++;
                } while (framegt.at<Vec3b>(l2,k2+1)[0] == 0   &&
                         framegt.at<Vec3b>(l2,k2+1)[1] == 255 &&
                         framegt.at<Vec3b>(l2,k2+1)[2] == 0);

                //Run down the side of the rectange to find bottom right corner
                do{
                   l2++;
                } while (framegt.at<Vec3b>(l2+1,k2)[0] == 0   &&
                         framegt.at<Vec3b>(l2+1,k2)[1] == 255 &&
                         framegt.at<Vec3b>(l2+1,k2)[2] == 0);


                //Double check that pixel is bottom right corner
                if ((framegt.at<Vec3b>(l2,k2+1)[0] != 0) &&
                    (framegt.at<Vec3b>(l2,k2+1)[1] != 255) &&
                    (framegt.at<Vec3b>(l2,k2+1)[2] != 0) &&
                    (framegt.at<Vec3b>(l2,k2-1)[0] == 0) &&
                    (framegt.at<Vec3b>(l2,k2-1)[1] == 255) &&
                    (framegt.at<Vec3b>(l2,k2-1)[2] == 0)){

                   //Check if bottom right corner of detection frame is within range of GT
                   for (int k3 = x_bot-x_tol; k3 <= x_bot+x_tol; k3++){
                      for (int l3 = y_bot-y_tol; l3 <= y_bot+y_tol; l3++){
                         if((l2==l3) && (k2==k3)){
                               std::cout << "True Positive Detection" << std::endl;
                               rectangle(frame, Point(x_top, y_top), Point(x_bot, y_bot), Scalar(0, 0, 255), 2); //For a true positive overwrite turquoise rectangular frame with red
                         }
                      }
                   }
                }
             }
          }
       }
    }
}
