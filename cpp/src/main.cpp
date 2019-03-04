#include <opencv2/opencv.hpp>
#include <iostream>
#include <sl/Camera.hpp>
#define SQUARE 21.1
#define scale 1.7

using namespace cv;
using namespace std;
using namespace sl;

cv::Mat slMat2cvMat(sl::Mat&);  //function declaration

int main(int argc, char* argv[]) {
	Size boardSize = Size(4, 11);    //interior corner
	Camera zed;
	InitParameters init_params;
	init_params.camera_resolution = RESOLUTION_HD1080;
	init_params.camera_fps = 30;
	init_params.coordinate_units = UNIT_MILLIMETER;
	init_params.camera_disable_self_calib = false;	//default value is also false
	ERROR_CODE err = zed.open(init_params);
	if (err != SUCCESS) {
		exit(-1);
	}

	sl::RuntimeParameters runtime_param;
	runtime_param.sensing_mode = SENSING_MODE_STANDARD;

	sl::Mat image_zed; //zed Mat
	cv::Mat image_ocv; //openCV Mat
    int j = 0;

	while (1) {
		if (zed.grab(runtime_param) == SUCCESS) {
			vector<Point3f> object_points;	//Although I just need one point 
			vector<Point2f> corners;	//save corners' pixel coordinates
            cin.clear();
            cin.sync();
			zed.retrieveImage(image_zed, VIEW_LEFT, MEM_CPU);   //the image_zed slMat2cvMat
			cv::Mat image_ocv = slMat2cvMat(image_zed);
            cv::Mat image_gray;
            cvtColor(image_ocv, image_gray, COLOR_BGR2GRAY); //source image to gray
            cv::Mat image_downSize;
            resize(image_ocv, image_downSize, Size(image_ocv.cols/scale, image_ocv.rows/scale), 0.0, 0.0, INTER_LINEAR);
            cv::Mat image_downGray; //change to Gray after downSize
            cvtColor(image_downSize, image_downGray, COLOR_BGR2GRAY);
            //imshow("resize", image_downSize);
			//unsigned long long timestamp = zed.getTimestamp(TIME_REFERENCE_IMAGE);
			bool found = findCirclesGrid(image_downGray,
                    boardSize,
                    corners,
                    CALIB_CB_ASYMMETRIC_GRID
                    );
            sl::Mat pointCloud;
            zed.retrieveMeasure(pointCloud, MEASURE_XYZRGBA, MEM_GPU);
            /*
            size_t pointCloud_Width = pointCloud.getWidth();
            size_t pointCloud_Height = pointCloud.getHeight();
            cout << "pointCloud_Width: " << pointCloud_Width;
            cout << "   pointCloud_Height: " << pointCloud_Height << endl;
			cout << "image width: "  << image_zed.getWidth();
			cout << "   image height: " << image_zed.getHeight() << endl;
            sl:float4 pcValue;     //point's depth value
            pointCloud.getValue<sl::float4>(pointCloud_Width/2, pointCloud_Height/2, &pcValue);   //getting depth information based on the image's pixel coordinates
            if(isnormal(pcValue.z))
              cout << pcValue.x << ", " << pcValue.y << ", " << pcValue.z << endl;
           */ 

			if (found) {
                for(int i=0; i<corners.size(); i++){
                    corners[i].x *= scale;
                    corners[i].y *= scale;
                }
                TermCriteria criteria = TermCriteria(
			    cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
					30,
					0.001);
				//cornerSubPix(image_gray, corners, Size(5, 5), Size(-1, -1), criteria);
                //cv::Mat image_gray;
                //cvtColor(image_ocv, image_gray, COLOR_BGR2GRAY);
				//find4QuadCornerSubpix(image_gray, corners, Size(5, 5)); 

                drawChessboardCorners(image_ocv, boardSize, corners, found);
			
                for(int i=0; i< boardSize.height; i++)
                    for (int j = 0; j < boardSize.width; j++) {
                        object_points.push_back(Point3f((float)((i%2+j*2)*SQUARE), (float)(i*SQUARE), 0.f));
                }
                sl::float3 pcValue;
                pointCloud.getValue<sl::float3>((int)(corners[0].x), (int)(corners[0].y), &pcValue);	//return the first corner's spatial coordinates(Relative to the ZED camera coordinate system)
                cout << "( " << pcValue.x << ", " << pcValue.y << ", " << pcValue.z << ")"<< endl;
                CalibrationParameters zed_StereoParam = zed.getCameraInformation().calibration_parameters;
                cv::Mat rvec, tvec;
                double *pDistortion = zed_StereoParam.left_cam.disto;
                cv::Mat distortion = (cv::Mat_<double>(5, 1) << pDistortion[0], pDistortion[1], pDistortion[2], pDistortion[3], pDistortion[4]);
                cv::Mat leftCameraIntrinic = (cv::Mat_<double>(3, 3) << zed_StereoParam.left_cam.fx, 0, zed_StereoParam.left_cam.cx, 0, zed_StereoParam.left_cam.fy, zed_StereoParam.left_cam.cy, 0, 0, 1);
                cv::solvePnP(object_points, corners, leftCameraIntrinic, distortion, rvec, tvec);
                cv::Mat Rotation3Mat;
                Rodrigues(rvec, Rotation3Mat);  //from Vector to Mat
                cout << tvec << "	" << endl;
                cout << Rotation3Mat << endl;
                printf("%d\n", j++);
			}//if checked corner,execute above
            imshow("picture", image_ocv);
            if (27 == waitKey(5))
                return 0;
		}//if grabed the image
	}//while()
}

cv::Mat slMat2cvMat(sl::Mat& input) {
	// Mapping between MAT_TYPE and CV_TYPE
	int cv_type = -1;
	switch (input.getDataType()) {
	case MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
	case MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
	case MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
	case MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
	case MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
	case MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
	case MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
	case MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
	default: break;
	}

	// Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
	// cv::Mat and sl::Mat will share a single memory structure
	return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
}
