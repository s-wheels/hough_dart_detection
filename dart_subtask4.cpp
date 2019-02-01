// header inclusion
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;


Mat Convolve(float kernel[3][3], Mat grayImg); //Convolution of the image
Mat Magnitude(Mat dx, Mat dy);                 //Magnitude of gradient of image
Mat Direction(Mat dx, Mat dy);                 //Direction of gradient of image
Mat Threshold(Mat magImg, int Thresh);         //Thresholds the magnitude image
pair<Mat, Mat> Hough(Mat threshMagImg, Mat dirImg, vector<Point> &circle_center, vector<int> &radius, int minRad, int maxRad, int threshold_c, int min_center_dist);//Hough space
Mat Centers(Mat houghPlot, int circleThresh, int minDist);  //Finds centre of circle according to certain threshold
Mat Circles(Mat houghSpace, Mat centerPlot, int minRad);    //Produces circles on black background
Mat Overlay(Mat img, Mat circlePlot, string colour);                       //Overlays circles on original image
tuple<Mat,Mat,Mat> ViolaJones(Mat img, Mat img_gt, int i, Mat hg_frame, float hough_tol, float gt_tol);             //Uses Viola-Jones method and compares with Hough Circle transform
array<int, 8> OutOfBounds(Mat img, int x1_top, int x2_bot, int x_tol, int y1_top, int y2_bot, int y_tol);   //Calculates bounds of search when comparing methods/groundtruth
int TruePositive(Mat tp_frame, Mat img_gt, int x1_top, int x2_bot, float x_tol, int y1_top, int y2_bot, float y_tol);  //Calculates whether detection can be considered True Positive or False by using groundtruth
Mat EmptyFrame(Mat img);

pair<Mat,Mat> intersection_point(Mat threshMagImg, Mat intersectionSpace, int threshold);
Mat hough_line(Mat threshMagImg, Mat dirImg, int error);
void linePoints(Mat threshMagImg, Mat houghLine, vector<Point> &point1, vector<Point> &point2, int threshold);
Mat lineIntersection(Mat threshMagImg, vector<Point> point1, vector<Point> point2);
Mat CentreComparison(Mat img, Mat intersec_output, vector<Point> &circle_center, vector<int> &radius, int li_tol);


//Load chosen cascade
string cascade_string = "cascade1";
string cascade_name = cascade_string + ".xml";
CascadeClassifier cascade;


int main()
{
 for (int i = 0; i < 16; i++)
 {

   //Determine particular dart image and the corresponding image with groundtruth markings
   string filename = "dart";
   filename = filename + to_string(i) + ".jpg";
   string filename_gt = "dartgtd";
   filename_gt = filename_gt + to_string(i) + ".png";

   Mat img = imread(filename);
   Mat img_gt = imread(filename_gt, CV_LOAD_IMAGE_COLOR);

   //Create border around images to ensure no incorrect array references later
   copyMakeBorder(img, img, 1, 1, 1, 1, BORDER_CONSTANT, 1);
   copyMakeBorder(img_gt, img_gt, 1, 1, 1, 1, BORDER_CONSTANT, 1);
   if(!cascade.load(cascade_name)){ printf("--(!)Cascade Error loading\n"); return -1; };

   //Create a gray-scale image of input
   Mat grayImg;
   cvtColor(img, grayImg, COLOR_RGB2GRAY);


  //Define differentiation kernels
  float dxkernel[3][3] =
   {
    { -1,0 ,1 },
    { -2,0 ,2 },
    { -1,0 ,1 }
   };

  float dykernel[3][3] =
  {
   { -1,-2,-1 },
   { 0 ,0 ,0 },
   { 1 ,2 ,1 }
  };

  //Calculate d/dx gradient of image and normalise values
  Mat dx = Convolve(dxkernel, grayImg);
  Mat dxnormal;
  normalize(dx, dxnormal, 0, 255, NORM_MINMAX, CV_8UC1);

  //Calculate d/dy gradient of image and normalise values
  Mat dy = Convolve(dykernel, grayImg);
  Mat dynormal;
  normalize(dy, dynormal, 0, 255, NORM_MINMAX, CV_8UC1);


  //Calculate both the gradient magnitude and direction and then normalise
  Mat magImg = Magnitude(dx, dy);
  Mat dirImg = Direction(dx, dy);
  Mat normDirImg;
  normalize(dirImg, normDirImg, 0, 255, NORM_MINMAX, CV_8UC1);

  //Threshold image so pixels above 50 are 255 and others are 0
  Mat threshMagImg = Threshold(magImg, 50);

  Mat houghLine = hough_line(threshMagImg, dirImg, 1);

  int lineThreshold = 7;
  vector<Point> point1;
  vector<Point> point2;
  linePoints(threshMagImg,houghLine, point1, point2,lineThreshold);



  Mat intersectionSpace = lineIntersection(threshMagImg, point1, point2);

  int threshLine = 3;
  pair<Mat,Mat> intersectionPoint = intersection_point(threshMagImg, intersectionSpace, threshLine);
  Mat intersecPoint = intersectionPoint.first;
  Mat intersec_output = intersectionPoint.second;;



  //check over different radii based on image
  int minRad = 10;
  int maxRad = 100;

  //Perform Hough Circle Transform on image
  vector<Point> circle_center;
  vector<int> radius;
  pair<Mat, Mat> hough = Hough(threshMagImg, dirImg, circle_center, radius, minRad, maxRad,10,50);
  Mat houghSpace = hough.first;
  Mat houghPlot = hough.second;
  normalize(houghPlot, houghPlot, 0, 255, NORM_MINMAX, CV_8UC1);


  //varying based on images int circleThresh = 100; int minDist = 100;
  int circleThresh = 100;
  int minDist = 100;
  Mat centerPlot = Centers(houghPlot, circleThresh, minDist);



  Mat hg_frame = Circles(houghSpace, centerPlot, minRad);
  Mat hg_overlay = Overlay(img, hg_frame, "Red");

  std::cout << "Press SPACE for next image" << std::endl;
  namedWindow("Hough Circle Detections", WINDOW_NORMAL);
  imshow("Hough Circle Detections", hg_overlay);
  waitKey(0);

  //Set tolerance value for max distance between Hough Line intersections and Hough circle centres
  int li_tol = 6;
  //Set tolerance value for V-J and Hough Circle detections being classified as the same dartboard
  float hough_tol = 0.55;
  //Set tolerance value based on detection img size for declaring whether TP or FP
  float gt_tol = 1.0;


  Mat lihg_frame = CentreComparison(img, intersec_output, circle_center, radius, li_tol);
  Mat lihg_overlay = Overlay(img, lihg_frame, "Blue");

  namedWindow("Hough Line and Circle Detections", WINDOW_NORMAL);
  imshow("Hough Line and Circle Detections", lihg_overlay);
  waitKey(0);

  //Viola-Jones with Hough circle centres and line intersections
  std::cout << "Viola-Jones, Hough Circles and Hough Line Intersections" << std::endl;
  tuple<Mat,Mat,Mat> frames_li = ViolaJones(img, img_gt, i, lihg_frame, hough_tol, gt_tol);
  Mat tpli_frame = get<0>(frames_li);
  Mat vjhgli_frame = get<1>(frames_li);
  Mat vj_frame = get<2>(frames_li);

  Mat vj_overlay = Overlay(img, vj_frame, "Yellow");
  namedWindow("Viola-Jones Detections", WINDOW_NORMAL);
  imshow("Viola-Jones Detections", vj_overlay);
  waitKey(0);

  Mat vjhgli_overlay = Overlay(img, vjhgli_frame, "Yellow");
  namedWindow("V-J and Hough Line/Circle Detections", WINDOW_NORMAL);
  imshow("V-J and Hough Line/Circle Detections", vjhgli_overlay);
  waitKey(0);

  //Viola-Jones with hough circle centres only
  std::cout << "Viola-Jones and Hough Circles" << std::endl;
  tuple<Mat,Mat,Mat> frames_hg = ViolaJones(img, img_gt, i, hg_frame, hough_tol, gt_tol);
  Mat tphg_frame = get<0>(frames_hg);
  Mat vjhg_frame = get<1>(frames_hg);

  Mat vjhg_overlay = Overlay(img, vjhg_frame, "Pink");
  namedWindow("V-J and Hough Circle Detections", WINDOW_NORMAL);
  imshow("V-J and Hough Circle Detections", vjhg_overlay);
  waitKey(0);


  Mat tphg_overlay = Overlay(img, tphg_frame, "Green");
  // namedWindow("True Positive Detections", WINDOW_NORMAL);
  // imshow("True Positive Detections", tphg_overlay);
  // waitKey(0);

  Mat tpli_overlay = Overlay(img, tpli_frame, "Green");
  // namedWindow("True Positive Detections", WINDOW_NORMAL);
  // imshow("True Positive Detections", tpli_overlay);
  // waitKey(0);

  img = Overlay(vjhg_overlay, tpli_frame, "Red");

  string output = "DartDetectOutput";
  output = output + to_string(i) + cascade_string + ".jpg";
  imwrite(output, img);



  }
  return 0;

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

tuple<Mat,Mat,Mat> ViolaJones(Mat img, Mat img_gt, int i, Mat hg_frame, float hough_tol, float gt_tol){

    std::vector<Rect> darts;
    Mat gray;

    // 1. Prepare Image by turning it into Grayscale and normalising lighting
    cvtColor(img, gray, CV_BGR2GRAY);
    equalizeHist(gray, gray);

    // 2. Perform Viola-Jones Object Detection
    cascade.detectMultiScale(gray, darts, 1.05, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50,50), Size(500,500));

    int size = darts.size();
    int t_pos;
    // 3. Print number of darts found by the detector
    std::cout << "Image = " << i << " | V-J Dartboards = " << size << std::endl;

    Mat vjhg_frame = EmptyFrame(img);
    Mat tp_frame = EmptyFrame(img);
    Mat vj_frame = EmptyFrame(img);

    //Run over all detections //d < size (below)
    for (int d = 0; d < size; d++){


      int x1_top = darts[d].x;
      int y1_top = darts[d].y;
      int x2_bot = darts[d].x + darts[d].width;
      int y2_bot = darts[d].y + darts[d].height;

      rectangle(vj_frame, Point(x1_top, y1_top), Point(x2_bot, y2_bot), 255, 2);



      int x_tol_hough = floor(hough_tol*darts[d].width);
      int y_tol_hough = floor(hough_tol*darts[d].height);

      array<int, 8>  bounds = {OutOfBounds(vj_frame, x1_top, x2_bot, x_tol_hough, y1_top, y2_bot, y_tol_hough)};


        //Loop over certain area around top right detection corner
      for (int n1 = bounds.at(0); n1 <= bounds.at(1); n1++){
        for (int m1 = bounds.at(2); m1 <= bounds.at(3); m1++){
            //Determine if pixel is top left corner of groundtruth

          if((hg_frame.at<uchar>(m1,n1)   == 255) &&
             (hg_frame.at<uchar>(m1+1,n1) == 255) &&
             (hg_frame.at<uchar>(m1,n1+1) == 255)){
            int n2 = n1;
            int m2 = m1;

            //Run along the top of the rectangle
            do{
              n2++;
            } while (hg_frame.at<uchar>(m2,n2+1) == 255);

             //Run down the side of the rectange to find bottom right corner
            do{
              m2++;
            } while (hg_frame.at<uchar>(m2+1,n2)== 255);

            //Double check that pixel is bottom right corner
            if (hg_frame.at<uchar>(m2,n2-1) == 255){

              //Check if bottom right corner of detection img is within range of GT
              for (int n3 = bounds.at(4); n3 <= bounds.at(5); n3++){
                 for (int m3 = bounds.at(6); m3 <= bounds.at(7); m3++){
                    if((m2==m3) && (n2==n3)){  //If bottom right corner of detection is within range of groundtruth bottom right corner then indicate it is true positive

                      rectangle(vjhg_frame, Point(x1_top, y1_top), Point(x2_bot, y2_bot), Scalar(255, 255, 0), 2);
                      int x_tol_gt = floor(gt_tol*darts[d].width);
                      int y_tol_gt = floor(gt_tol*darts[d].height);
                      t_pos = TruePositive(tp_frame, img_gt, x1_top, x2_bot, x_tol_gt, y1_top, y2_bot, y_tol_gt);
                  }
                }
              }
            }
          }
        }
      }
    }
    std::cout << "True Positives = " << t_pos << " | False Positives = " << size-t_pos << std::endl;
    std::cout << " " << std::endl;
    return make_tuple(tp_frame,vjhg_frame,vj_frame);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

array<int, 8> OutOfBounds(Mat img, int x1_top, int x2_bot, int x_tol, int y1_top, int y2_bot, int y_tol){

  int x1_beg, x1_end, y1_beg, y1_end, x2_beg, x2_end, y2_beg, y2_end;

  //If statements ensuring no 'out of bounds' array references are made
  if(x1_top-x_tol < 0){
    x1_beg = 0;
  }else{
    x1_beg = x1_top-x_tol;
  }

  if(x1_top+x_tol > img.cols){
    x1_end = img.cols;
  }else{
    x1_end = x1_top + x_tol;
  }

  if(y1_top-y_tol < 0){
    y1_beg = 0;
  }else{
    y1_beg = y1_top-y_tol;
  }

  if(y1_top+y_tol > img.rows){
    y1_end = img.rows;
  }else{
    y1_end = y1_top + y_tol;
  }

  if(x2_bot-x_tol < 0){
    x2_beg = 0;
  }else{
    x2_beg = x2_bot-x_tol;
  }

  if(x2_bot+x_tol > img.cols){
    x2_end = img.cols;
  }else{
    x2_end = x2_bot + x_tol;
  }

  if(y2_bot-y_tol < 0){
    y2_beg = 0;
  }else{
    y2_beg = y2_bot-y_tol;
  }

  if(y2_bot+y_tol > img.rows){
    y2_end = img.rows;
  }else{
    y2_end = y2_bot + y_tol;
  }

  array< int, 8 > bounds = {x1_beg, x1_end, y1_beg, y1_end, x2_beg, x2_end, y2_beg, y2_end};

  return bounds;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int TruePositive(Mat tp_frame, Mat img_gt, int x1_top, int x2_bot, float x_tol, int y1_top, int y2_bot, float y_tol){

  int t_pos = 0;       //True positive detection counter


  array<int, 8>  bounds = {OutOfBounds(img_gt, x1_top, x2_bot, x_tol, y1_top, y2_bot, y_tol)};

  //Loop over certain area around top right detection corner
  for (int k1 = bounds.at(0); k1 <= bounds.at(1); k1++){
     for (int l1 = bounds.at(2); l1 <= bounds.at(3); l1++){
        //Determine if pixel is top left corner of groundtruth

        if ((img_gt.at<Vec3b>(l1,k1)[0]   == 0) &&
            (img_gt.at<Vec3b>(l1,k1)[1]   == 255) &&
            (img_gt.at<Vec3b>(l1,k1)[2]   == 0) &&
            (img_gt.at<Vec3b>(l1+1,k1)[0] == 0) &&
            (img_gt.at<Vec3b>(l1+1,k1)[1] == 255) &&
            (img_gt.at<Vec3b>(l1+1,k1)[2] == 0) &&
            (img_gt.at<Vec3b>(l1,k1+1)[0] == 0) &&
            (img_gt.at<Vec3b>(l1,k1+1)[1] == 255) &&
            (img_gt.at<Vec3b>(l1,k1+1)[2] == 0)){
           int k2 = k1;
           int l2 = l1;

           //Run along the top of the rectangle
           do{
              k2++;
           } while (img_gt.at<Vec3b>(l2,k2+1)[0] == 0   &&
                    img_gt.at<Vec3b>(l2,k2+1)[1] == 255 &&
                    img_gt.at<Vec3b>(l2,k2+1)[2] == 0);

             //Run down the side of the rectange to find bottom right corner
           do{
              l2++;
           } while (img_gt.at<Vec3b>(l2+1,k2)[0] == 0   &&
                    img_gt.at<Vec3b>(l2+1,k2)[1] == 255 &&
                    img_gt.at<Vec3b>(l2+1,k2)[2] == 0);

           //Double check that pixel is bottom right corner
           if ((img_gt.at<Vec3b>(l2,k2-1)[0] == 0) &&
               (img_gt.at<Vec3b>(l2,k2-1)[1] == 255) &&
               (img_gt.at<Vec3b>(l2,k2-1)[2] == 0)){

              //Check if bottom right corner of detection img is within range of GT
              for (int k3 = bounds.at(4); k3 <= bounds.at(5); k3++){
                 for (int l3 = bounds.at(6); l3 <= bounds.at(7); l3++){
                    if((l2==l3) && (k2==k3)){  //If bottom right corner of detection is within range of groundtruth bottom right corner then indicate it is true positive
                       t_pos += 1;
                       rectangle(tp_frame, Point(x1_top, y1_top), Point(x2_bot, y2_bot), 255, 2); //For a true positive write rectangular frame
                    }
                 }
              }
           }
        }
     }
  }
  return t_pos;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Mat Convolve(float kernel[3][3], Mat grayImg) {

  Mat grayFilteredImg = grayImg.clone();

  //Convert 'gray' uchar image to 32bit float image
  grayFilteredImg.convertTo(grayFilteredImg, CV_32F);

  //For particular point in input image
  for (int y = 1; y<grayImg.rows - 1; y++) {
    for (int x = 1; x<grayImg.cols - 1; x++) {

      grayFilteredImg.at<float>(y, x) = 0.0;

      //Loops over kernel elements
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          //Convolutes img with kernel
          grayFilteredImg.at<float>(y, x) = (grayFilteredImg.at<float>(y, x) + float(grayImg.at<uchar>(y + i, x + j) * kernel[i + 1][j + 1]));

        }
      }

    }
  }

  return grayFilteredImg;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Mat Magnitude(Mat dx, Mat dy) {

  Mat magImg = dx.clone();

  //Loop over entire img and calculate the combined magnitude of dx and dy
  for (int y = 0; y < magImg.rows; y++) {
    for (int x = 0; x < magImg.cols; x++) {
      magImg.at<float>(y, x) = 0.0;
      magImg.at<float>(y, x) = pow(pow(float(dx.at<float>(y, x)), 2.0) + pow(float(dy.at<float>(y, x)), 2.0), 0.5);
    }
  }

  //Normalise image and convert to 8bit uchar
  normalize(magImg, magImg, 0, 255, NORM_MINMAX, CV_8UC1);
  return magImg;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Mat Direction(Mat dx, Mat dy) {

  Mat dirImg = dx.clone();
  //Loop over entire img and calculate the direction of the gradient by combining d(img)/dx and d(img)/dy
  for (int y = 0; y < dirImg.rows; y++) {
    for (int x = 0; x < dirImg.cols; x++) {
      dirImg.at<float>(y, x) = 0.0;
      dirImg.at<float>(y, x) = float((atan2(double(dy.at<float>(y, x)), double(dx.at<float>(y, x)))));
    }
  }
  return dirImg;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Mat Threshold(Mat magImg, int Thresh) {

  //Loop over entire image and threshold it so points with pixel val above threshold are 255 and others are set to 0
  for (int y = 0; y < magImg.rows; y++) {
    for (int x = 0; x < magImg.cols; x++) {
      if (magImg.at<uchar>(y, x) >= Thresh) {
        magImg.at<uchar>(y, x) = 255;
      }
      else {
        magImg.at<uchar>(y, x) = 0;
      }
    }
  }

  return magImg;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Mat hough_line(Mat threshMagImg, Mat dirImg, int error) {

  //rho is distance to line, rhoMax is maximum distance to line
	float rhoMax = sqrt(threshMagImg.rows*threshMagImg.rows + threshMagImg.cols*threshMagImg.cols);
  int gradientAngle, rho, rhoHough;


  Mat outputHough;
  outputHough.create(rhoMax * 2, 180, CV_32F);
  for (int j = 0; j < outputHough.rows; j++) {
    for (int i = 0; i < outputHough.cols; i++) {
      outputHough.at<float>(j, i) = 0;
    }
  }

	// Line hough space
	for (int y = 0; y < threshMagImg.rows; y++) {
		for (int x = 0; x < threshMagImg.cols; x++) {
      //If point is an edge (thresholded)
			if (threshMagImg.at<uchar>(y, x) == 255) {
        //Check for all angles
				for (int theta = 0; theta < 180; theta++) {

					gradientAngle = cvRound(dirImg.at<float>(y, x) * 180 / CV_PI);
          //If angle of gradient is within error bounds of theta
					if (gradientAngle >= theta - error && gradientAngle <= theta + error) {
            //Equation of line and vote in Hough space
            rho = cvRound(x*cos(float(theta) * CV_PI / 180) + y*sin(float(theta) * CV_PI / 180));
						rhoHough = rho + rhoMax;
						outputHough.at<float>(rhoHough, theta) = outputHough.at<float>(rhoHough, theta) + 1;

					}
				}
			}
		}
	}
  return outputHough;
}

void linePoints(Mat threshMagImg, Mat houghLine, vector<Point> &point1, vector<Point> &point2, int threshold) {

	//Recover rho real value from hough line space
	float rhoMax = sqrt(threshMagImg.rows*threshMagImg.rows + threshMagImg.cols*threshMagImg.cols);
  Point p1,p2;
  float c,s,rho,theta;

  //Across the Hough Line Space
	for (int j = 0; j < houghLine.rows; j++) {
		for (int i = 0; i < houghLine.cols; i++) {

			//thresholding
			if (houghLine.at<float>(j, i) > threshold) {

				//Obtain 2 points of each line using Hough Line Parameters
        rho = j - rhoMax;
				theta = i * CV_PI / 180;
				c = cos(theta);
        s = sin(theta);

				double x0 = c*rho, y0 = s*rho;
				p1.x = cvRound(x0 + 10 * (-s));
				p1.y = cvRound(y0 + 10 * (c));

        point1.push_back(Point(p1.x, p1.y));

				p2.x = cvRound(x0 - 10 * (-s));
				p2.y = cvRound(y0 - 10 * (c));

				point1.push_back(Point(p1.x, p1.y));
				point2.push_back(Point(p2.x, p2.y));
			}
		}
	}
}

Mat lineIntersection(Mat threshMagImg, vector<Point> point1, vector<Point> point2) {

	Mat intersectionSpace;
  float m1, m2, c1, c2;
  float intersectionY, intersectionX;
  int Yrounded, Xrounded;

	// Initialize intersection space
	intersectionSpace.create(threshMagImg.size(), threshMagImg.type());
	for (int y = 0; y < threshMagImg.rows; y++) {
		for (int x = 0; x < threshMagImg.cols; x++) {
			intersectionSpace.at<uchar>(y, x) = 0;
		}
  }

	for (int i = 0; i < point1.size(); i++)
  {
    // slope and intercept of a line with 2 non-identical points
    if (point2[i].x != point1[i].x)
    {
      //Calculate parameters in y=mx+c
      m1 = (float(point2[i].y - point1[i].y)) / (point2[i].x - point1[i].x);
			c1 = point1[i].y - (float((point2[i].y - point1[i].y)) / (point2[i].x - point1[i].x))*point1[i].x;

      //slope and intercept of the following points and the corresponding lines
			for (int j = i + 1; j < point1.size(); j++)
      {
				if ((point2[j].x != point1[j].x))
        {
          //Calculate parameters in y=mx+c
          m2 = float(point2[j].y - point1[j].y) / (point2[j].x - point1[j].x);
					c2 = point1[j].y - (float(point2[j].y - point1[j].y) / (point2[j].x - point1[j].x)) * point1[j].x;

					intersectionY = ((m1*m2 - c1*c2) / (m1 - m2));
          Yrounded = cvRound(intersectionY);

					intersectionX = ((c2 - c1) / (m1 - m2));
          Xrounded = cvRound(intersectionX);

          //If the lines are not the same and the points are real
					if (m1 != m2 && Yrounded > 0 && Yrounded < intersectionSpace.rows && Xrounded > 0 && Xrounded < intersectionSpace.cols)
          {
            //Vote in intersection space at intersection
						intersectionSpace.at<uchar>(Yrounded, Xrounded) = intersectionSpace.at<uchar>(Yrounded, Xrounded) + 1;
					}
				}
			}
		}
	}

	return intersectionSpace;
}

pair<Mat,Mat> intersection_point(Mat threshMagImg, Mat intersectionSpace, int threshold) {

  Mat outputIntersection = threshMagImg.clone();
  Mat outputImg;
	cvtColor(threshMagImg, outputImg, CV_GRAY2BGR);

	for (int y = 0; y < intersectionSpace.rows; y++) {
		for (int x = 0; x < intersectionSpace.cols; x++) {
			if (intersectionSpace.at<uchar>(y, x) > threshold) {
				circle(outputImg, Point(x, y), 1, Scalar(255, 0, 0), 1);
				outputIntersection.at<uchar>(y, x) = 255;
			}
		}
	}
	return make_pair(outputImg,outputIntersection);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// pair<Mat,Mat> Hough(Mat threshMagImg, Mat dirImg, int minRad, int maxRad, int threshold_c, int min_center_dist) {
//
// 	double max;
// 	vector<Point> circle_center; // location of centres
// 	vector<int> circle_r;
// 	vector<int> accum; // accumulator value for each centre
//
//
//   int sizesSpace[] = { threshMagImg.rows, threshMagImg.cols, maxRad - minRad + 1 };
// 	int sizesPlot[] = { threshMagImg.rows, threshMagImg.cols };
//
// 	Mat houghSpace = Mat(3, sizesSpace, CV_32F, Scalar(0));
// 	Mat houghPlot = Mat(2, sizesPlot, CV_32F, Scalar(0));
//
// 	for (int r = minRad; r <= maxRad; r ++) {
//
//                 Mat initial_mat = Mat::zeros(threshMagImg.size(), CV_8UC1);
//
//                 for (int y = 0; y < threshMagImg.rows; y++) {
//                         for (int x = 0; x < threshMagImg.cols; x++) {
//
//
//                                 if (threshMagImg.at<uchar>(y,x) == 255) {
//                                         for (int i = 1; i <= 4; i++) {
//
//                                                 int x0 = 0;
//                                                 int y0 = 0;
//
//                                                 switch (i) {
//                                                 case 1:
//                                                         x0 = int(float(x) + (float(r) * float(cos(dirImg.at<float>(y, x) + 3.14/2)))); //polar coordinate for cente
//                                                         y0 = int(float(y) + (float(r) * float(sin(dirImg.at<float>(y, x) + 3.14/2))));
//                                                         break;
//                                                 case 2:
//                                                         x0 = int(float(x) + (float(r) * float(cos(dirImg.at<float>(y, x) + 3.14/2)))); //polar coordinate for cente
//                                                         y0 = int(float(y) - (float(r) * float(sin(dirImg.at<float>(y, x) + 3.14/2))));
//                                                         break;
//                                                 case 3:
//                                                         x0 = int(float(x) - (float(r) * float(cos(dirImg.at<float>(y, x) + 3.14/2)))); //polar coordinate for cente
//                                                         y0 = int(float(y) + (float(r) * float(sin(dirImg.at<float>(y, x) + 3.14/2))));
//                                                         break;
//                                                 case 4:
//                                                         x0 = int(float(x) - (float(r) * float(cos(dirImg.at<float>(y, x) + 3.14/2)))); //polar coordinate for cente
//                                                         y0 = int(float(y) - (float(r) * float(sin(dirImg.at<float>(y, x) + 3.14/2))));
//                                                         break;
//                                                 }
//
//                                                 if ((y > r) && (y < int(threshMagImg.rows - r)) && (x > r) && (x < int(threshMagImg.cols - r)))
//                                                         initial_mat.at<uchar>(y0, x0) = initial_mat.at<uchar>(y0, x0) + 1;
//
//
//                                                 if ((0 < x0 && x0 < int(threshMagImg.cols - 1)) && (0 < y0 && y0 < int(threshMagImg.rows - 1))) {
//                                                         houghSpace.at<float>(y0, x0, r - minRad)++;
//                                                         houghPlot.at<float>(y0, x0)++;
//                                                 }
//
//
//                                         }
//
//                                 }
//
// 			}
// 		}
//
//                 // Maximum value extract
// 		cv::minMaxLoc(initial_mat, NULL, &max);
//                 // Location of the maximum value
// 		cv::Point max_loc;
// 		cv::minMaxLoc(initial_mat, NULL, &max, NULL, &max_loc);
//
// 		// Threshold
// 		if (max > threshold_c) {
//       // adding the location and the value of the max vector to the circle_center vector and the accumulator
// 			circle_center.push_back(Point(max_loc));
// 			circle_r.push_back(r);
// 			accum.push_back(cvRound(max));
// 		}
//
//
// 	}
//
//
//
// 	// Pick max value for R and (x,y) after running through every pixel
// 	for (int i = 0; i < circle_r.size(); i++) {
// 		for (int j = i + 1; j < circle_r.size(); j++) {
//
// 			// compare the centres and discard the lower value centres in case of overlap
// 			if ( (circle_center[i].x > circle_center[j].x - min_center_dist) && (circle_center[i].x < circle_center[j].x + min_center_dist)
// 				&& (circle_center[i].y > circle_center[j].y - min_center_dist) && (circle_center[i].y < circle_center[j].y + min_center_dist) ) {
//
// 				//accum val[i] > accum val[j] ?
// 				if (accum[i] > accum[j]) {
//
// 					circle_center.erase(circle_center.begin() + j);
// 					circle_r.erase(circle_r.begin() + j);
// 					accum.erase(accum.begin() + j);
// 					j--;
//
// 				}
//
// 				// accum val[i] <= accum val[j] ?
// 				else {
// 					circle_center.erase(circle_center.begin() + i);
// 					circle_r.erase(circle_r.begin() + i);
// 					accum.erase(accum.begin() + i);
//
// 						i--;
// 				}
// 			}
// 		}
// 	}
//
// 	return make_pair(houghSpace,houghPlot);
// }

pair<Mat,Mat> Hough(Mat threshMagImg, Mat dirImg, vector<Point> &circle_center, vector<int> &radius, int minRad, int maxRad, int threshold_c, int min_center_dist) {

	double max;
	//vector<Point> circle_center; // location of centres
	//vector<int> circle_r;
	vector<int> accum; // accumulator value for each centre


  int sizesSpace[] = { threshMagImg.rows, threshMagImg.cols, maxRad - minRad + 1 };
	int sizesPlot[] = { threshMagImg.rows, threshMagImg.cols };

	Mat houghSpace = Mat(3, sizesSpace, CV_32F, Scalar(0));
	Mat houghPlot = Mat(2, sizesPlot, CV_32F, Scalar(0));

	for (int r = minRad; r <= maxRad; r ++) {

                Mat initial_mat = Mat::zeros(threshMagImg.size(), CV_8UC1);

                for (int y = 0; y < threshMagImg.rows; y++) {
                        for (int x = 0; x < threshMagImg.cols; x++) {


                                if (threshMagImg.at<uchar>(y,x) == 255) {
                                        for (int i = 1; i <= 4; i++) {

                                                int x0 = 0;
                                                int y0 = 0;

                                                switch (i) {
                                                case 1:
                                                        x0 = int(float(x) + (float(r) * float(cos(dirImg.at<float>(y, x)+3.14/2)))); //polar coordinate for centre
                                                        y0 = int(float(y) + (float(r) * float(sin(dirImg.at<float>(y, x)+3.14/2))));
                                                        break;
                                                case 2:
                                                        x0 = int(float(x) + (float(r) * float(cos(dirImg.at<float>(y, x)+3.14/2)))); //polar coordinate for centre
                                                        y0 = int(float(y) - (float(r) * float(sin(dirImg.at<float>(y, x)+3.14/2))));
                                                        break;
                                                case 3:
                                                        x0 = int(float(x) - (float(r) * float(cos(dirImg.at<float>(y, x)+3.14/2)))); //polar coordinate for centre
                                                        y0 = int(float(y) + (float(r) * float(sin(dirImg.at<float>(y, x)+3.14/2))));
                                                        break;
                                                case 4:
                                                        x0 = int(float(x) - (float(r) * float(cos(dirImg.at<float>(y, x)+3.14/2)))); //polar coordinate for centre
                                                        y0 = int(float(y) - (float(r) * float(sin(dirImg.at<float>(y, x)+3.14/2))));
                                                        break;
                                                }

                                                if ((y > r) && (y < int(threshMagImg.rows - r)) && (x > r) && (x < int(threshMagImg.cols - r)))
                                                        initial_mat.at<uchar>(y0, x0) = initial_mat.at<uchar>(y0, x0) + 1;


                                                if ((0 < x0 && x0 < int(threshMagImg.cols - 1)) && (0 < y0 && y0 < int(threshMagImg.rows - 1))) {
                                                        houghSpace.at<float>(y0, x0, r - minRad)++;
                                                        houghPlot.at<float>(y0, x0)++;
                                                }
                                        }
                                }
			}
		}

    // Maximum value extract
		minMaxLoc(initial_mat, NULL, &max);
    // Location of the maximum value
		Point max_loc;
		minMaxLoc(initial_mat, NULL, &max, NULL, &max_loc);

		// Threshold
		if (max > threshold_c) {
                    // adding the location and the value of the max vector to the circle_center vector and the accumulator
			circle_center.push_back(Point(max_loc));
			radius.push_back(r);
			accum.push_back(cvRound(max));
		}


	}



	// Pick max value for R and (x,y) after running through every pixel
	for (int i = 0; i < radius.size(); i++) {
		for (int j = i + 1; j < radius.size(); j++) {

			// compare the centres and discard the lower value centres in case of overlap
			if ( (circle_center[i].x > circle_center[j].x - min_center_dist) && (circle_center[i].x < circle_center[j].x + min_center_dist)
				&& (circle_center[i].y > circle_center[j].y - min_center_dist) && (circle_center[i].y < circle_center[j].y + min_center_dist) ) {

				//accum val[i] > accum val[j] ?
				if (accum[i] > accum[j]) {

					circle_center.erase(circle_center.begin() + j);
					radius.erase(radius.begin() + j);
					accum.erase(accum.begin() + j);
					j--;

				}

				// accum val[i] <= accum val[j] ?
				else {
					circle_center.erase(circle_center.begin() + i);
					radius.erase(radius.begin() + i);
					accum.erase(accum.begin() + i);

						i--;
				}
			}
		}
	}

	return make_pair(houghSpace,houghPlot);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Mat Centers(Mat houghPlot, int circleThresh, int minDist){

  //Define an image for plotting centres
  Mat centerPlot = EmptyFrame(houghPlot);

  uchar maxPixelVal = 0;
  int maxPixelY;
  int maxPixelX;
  bool isEnd = false;


  //Finds maximum point of hough circle space
  while (isEnd == false)
  {
    //Iterate over entire real image (not border) and find highest pixel value and its location
    for (int y = 1; y < houghPlot.rows-1; y++)
    {
      for (int x = 1; x < houghPlot.cols-1; x++)
      {

        if (houghPlot.at<uchar>(y, x) > maxPixelVal)
        {
          maxPixelVal = houghPlot.at<uchar>(y, x);
          maxPixelY = y;
          maxPixelX = x;
        }
      }
    }

    if (maxPixelVal >= circleThresh)
    {
      centerPlot.at<uchar>(maxPixelY, maxPixelX) = 255;

      //Sets a circle of radius minDist around (maxPixelY, maxPixelX) to zero

      for (int y = max((maxPixelY - minDist), 0); y < min((maxPixelY + minDist), houghPlot.rows); y++)
      {
        for (int x = max((maxPixelX - minDist), 0); x < min((maxPixelX + minDist), houghPlot.cols); x++)
        {

          if (pow((y - maxPixelY), 2) + pow((x - maxPixelX), 2) <= pow(minDist, 2))
          {
            houghPlot.at<uchar>(y, x) = 0;
          }
        }
      }
      maxPixelVal = 0;
    }
    else
      isEnd = true;
  }

  return centerPlot;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Mat Circles(Mat houghSpace, Mat centerPlot, int minRad) {

  Mat circlePlot = EmptyFrame(houghSpace);

  // houghSpace.size[2] is the number of radii

  for (int y = 0; y < centerPlot.rows; y++) {
    for (int x = 0; x < centerPlot.cols; x++) {

      if (centerPlot.at<uchar>(y, x) == 255) {

        float Rad = 0.0;
        int maxHough = 0;

        for (int r = 0; r < houghSpace.size[2]; r++)
        {
          if (houghSpace.at<float>(y, x, r) > maxHough)
          {
            Rad = (float(r) + float(minRad));
            maxHough = houghSpace.at<float>(y, x, r);
          }
        }

        int x_tl = x-Rad;
        int y_tl = y-Rad;
        int x_br = x+Rad;
        int y_br = y+Rad;

        if (x_tl <= 0){
          x_tl = 1;
        }
        if (y_tl <= 0){
          y_tl = 1;
        }
        if (x_br >= centerPlot.cols){
          x_br = centerPlot.cols-1;
        }
        if (y_br >= centerPlot.rows-1){
          y_br = centerPlot.rows-1;
        }

        circle(circlePlot, Point(x,y), Rad, 200, 1, 8, 0);
        rectangle(circlePlot, Point(x_tl, y_tl), Point(x_br, y_br), 255, 1);

      }
    }

  }

  return circlePlot;

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Mat Overlay(Mat img, Mat det_frame, string colour) {

  Mat imgWithOverlay = img.clone();

  //Iterates over detection frames and transposes detections to coloured detections on original image
  for (int y = 1; y < img.rows-1; y++) {
    for (int x = 1; x < img.cols-1; x++) {

      if (colour=="Green"){
             if (det_frame.at<uchar>(y,x) == 255) //GREEN
             {
               imgWithOverlay.at<Vec3b>(y, x)[0] = 0;
               imgWithOverlay.at<Vec3b>(y, x)[1] = 255;
               imgWithOverlay.at<Vec3b>(y, x)[2] = 0;
             }
      }
      if (colour=="Red"){
             if (det_frame.at<uchar>(y,x) == 255) //RED
             {
               imgWithOverlay.at<Vec3b>(y, x)[0] = 0;
               imgWithOverlay.at<Vec3b>(y, x)[1] = 0;
               imgWithOverlay.at<Vec3b>(y, x)[2] = 255;
             }
      }
      if (colour=="Blue"){
             if (det_frame.at<uchar>(y,x) == 255) //BLUE
             {
               imgWithOverlay.at<Vec3b>(y, x)[0] = 255;
               imgWithOverlay.at<Vec3b>(y, x)[1] = 0;
               imgWithOverlay.at<Vec3b>(y, x)[2] = 0;
             }
      }
      if (colour=="Turquoise"){
             if (det_frame.at<uchar>(y,x) == 255) //TURQUOISE
             {
               imgWithOverlay.at<Vec3b>(y, x)[0] = 255;
               imgWithOverlay.at<Vec3b>(y, x)[1] = 255;
               imgWithOverlay.at<Vec3b>(y, x)[2] = 0;
             }
      }
      if (colour=="Yellow"){
             if (det_frame.at<uchar>(y,x) == 255) //YELLOW
             {
               imgWithOverlay.at<Vec3b>(y, x)[0] = 0;
               imgWithOverlay.at<Vec3b>(y, x)[1] = 255;
               imgWithOverlay.at<Vec3b>(y, x)[2] = 255;
             }
      }
      if (colour=="Pink"){
             if (det_frame.at<uchar>(y,x) == 255) //PINK
             {
               imgWithOverlay.at<Vec3b>(y, x)[0] = 255;
               imgWithOverlay.at<Vec3b>(y, x)[1] = 0;
               imgWithOverlay.at<Vec3b>(y, x)[2] = 255;
             }
      }
    }

  }
  return imgWithOverlay;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Mat EmptyFrame(Mat img) //Produces empty gray-scale image
{
  int ImgDims[] = { img.size[0], img.size[1] };
  Mat det_frame = Mat(2, ImgDims, CV_8U, Scalar(0));
  return det_frame;
}

Mat CentreComparison(Mat img, Mat intersec_output, vector<Point> &circle_center, vector<int> &radius, int li_tol)
{

  Mat lihg_frame = EmptyFrame(img);
  int counter;


	for (int i = 0; i < circle_center.size(); i++) {
		counter = 0;
    for (int j = -li_tol; j < li_tol; j++)
    {
			for (int k = -li_tol; k < li_tol; k++)
      {

				if (intersec_output.at<uchar>(Point(circle_center[i].x + k, circle_center[i].y + j)) == 255)
        {
					rectangle(lihg_frame, Point(circle_center[i].x - radius[i], circle_center[i].y - radius[i]),
					  Point(circle_center[i].x + radius[i], circle_center[i].y + radius[i]), 255, 2);
					counter++;
				}

				if (counter > 0)
					break;
			}
			if (counter > 0)
				break;
		}
	}

	return lihg_frame;
}
