#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    
    String modelConfiguration = "yolo.cfg";
    String modelBinary = "yolo.weights";
    
    //! [Initialize network]
    net = readNetFromDarknet(modelConfiguration, modelBinary);
    //! [Initialize network]
    if (net.empty())
    {
        cout << "Can't load network by using the following files: " << endl;
        cout << "cfg-file:     " << modelConfiguration << endl;
        cout << "weights-file: " << modelBinary << endl;
        cout << "Models can be downloaded here:" << endl;
        cout << "https://pjreddie.com/darknet/yolo/" << endl;
    }
    
    // cam
    video.setDeviceID( 0 );
    video.setDesiredFrameRate( 30 );
    video.initGrabber( 640, 480 );
    
    // img
    detectImg.load("my.jpg");
    getDetectedImageFromYOLO(detectImg.getPixels());
    detectImg.update();
    
    // objectClassName
    ifstream classNamesFile("coco.names");
    if (classNamesFile.is_open())
    {
        string className = "";
        while (std::getline(classNamesFile, className))
            classNamesVec.push_back(className);
        
        for( auto itr : classNamesVec )
        {
            string cName = itr;
            cout << "classNames :" << cName << endl;
        }
    }
}

//--------------------------------------------------------------
void ofApp::update(){
    video.update();
    getDetectedImageFromYOLO(video.getPixels());
    cameraImg = video.getPixels();
}

//--------------------------------------------------------------
void ofApp::draw(){
    ofSetColor(255);
    video.draw(0, 0);
    cameraImg.draw(640,0);
    
    detectImg.draw(0,480);
    
    ofDrawBitmapString(ofToString(ofGetFrameRate(), 0),20, 20);
}


cv::Mat ofApp::toCV(ofPixels &pix)
{
    return cv::Mat(pix.getHeight(), pix.getWidth(), CV_MAKETYPE(CV_8U, pix.getNumChannels()), pix.getData(), 0);
}

//--------------------------------------------------------------
void ofApp::getDetectedImageFromYOLO(ofPixels &op){
    cv::Mat frame = toCV(op);
    
    if (frame.channels() == 4)
        cvtColor(frame, frame, COLOR_BGRA2BGR);
    
    //! [Resizing without keeping aspect ratio]
    Mat resized;
    resize(frame, resized, cvSize(network_width, network_height));
    //! [Resizing without keeping aspect ratio]
    
    //! [Prepare blob]
    Mat inputBlob = blobFromImage(resized, 1 / 255.F); //Convert Mat to batch of images
    //! [Prepare blob]
    
    //! [Set input blob]
    net.setInput(inputBlob, "data");                   //set the network input
    //! [Set input blob]
    
    //! [Make forward pass]
    Mat detectionMat = net.forward("detection_out");   //compute output
    //! [Make forward pass]
    
    vector<double> layersTimings;
    double freq = getTickFrequency() / 1000;
    double time = net.getPerfProfile(layersTimings) / freq;
    ostringstream ss;
    ss << "FPS: " << 1000/time << " ; time: " << time << " ms";
    putText(frame, ss.str(), cvPoint(20,20), 0, 0.5, Scalar(0,0,255));
    
    float confidenceThreshold = 0.24;
    
    for (int i = 0; i < detectionMat.rows; i++)
    {
        const int probability_index = 5;
        const int probability_size = detectionMat.cols - probability_index;
        float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
        
        size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
        float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
        
        if (confidence > confidenceThreshold)
        {
            float x = detectionMat.at<float>(i, 0);
            float y = detectionMat.at<float>(i, 1);
            float width = detectionMat.at<float>(i, 2);
            float height = detectionMat.at<float>(i, 3);
            int xLeftBottom = static_cast<int>((x - width / 2) * frame.cols);
            int yLeftBottom = static_cast<int>((y - height / 2) * frame.rows);
            int xRightTop = static_cast<int>((x + width / 2) * frame.cols);
            int yRightTop = static_cast<int>((y + height / 2) * frame.rows);
            
            rectangle(frame, cvRect(xLeftBottom, yLeftBottom,xRightTop - xLeftBottom,yRightTop - yLeftBottom), Scalar(0, 255, 0));
            
            if (objectClass < classNamesVec.size())
            {
                ss.str("");
                ss << confidence;
                String conf(ss.str());
                String label = String(classNamesVec[objectClass]) + ": " + conf;
                int baseLine = 0;
                CvSize labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                rectangle(frame, cvRect(xLeftBottom, yLeftBottom ,labelSize.width, labelSize.height + baseLine),Scalar(255, 255, 255), CV_FILLED);
                putText(frame, label, cvPoint(xLeftBottom, yLeftBottom + labelSize.height),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
            }
            else
            {
                cout << "Class: " << objectClass << endl;
                cout << "Confidence: " << confidence << endl;
                cout << " " << xLeftBottom
                << " " << yLeftBottom
                << " " << xRightTop
                << " " << yRightTop << endl;
            }
        }
    }
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
