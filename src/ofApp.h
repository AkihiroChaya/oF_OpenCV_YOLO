#pragma once

#include "ofMain.h"

// YOLO
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>

using namespace std;
using namespace cv;
using namespace cv::dnn;

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
    
    dnn::Net net;
    
    int network_width = 416;
    int network_height = 416;
    
    ofVideoGrabber video;
    ofImage detectImg; // 静止画
    ofImage cameraImg; //　webCam
    
    vector<string> classNamesVec;
    
    cv::Mat toCV(ofPixels &pix);
    void getDetectedImageFromYOLO(ofPixels &op);
    
};
