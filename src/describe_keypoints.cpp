#include <iostream>
#include <fstream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;

void descKeypoints1()
{
    // Intialize output file
    ofstream outfile;
    outfile.open("../out/results.txt");
    if (!outfile.is_open())
    {
        cout << "Unable to open file" << endl;
        return;
    }


    // load image from file and convert to grayscale
    cv::Mat imgGray;
    cv::Mat img = cv::imread("../images/img1.png");
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // BRISK detector / descriptor
    cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
    vector<cv::KeyPoint> kptsBRISK;

    double t = (double)cv::getTickCount();
    detector->detect(imgGray, kptsBRISK);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK detector with n = " << kptsBRISK.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    outfile << "BRISK detector with n = " << kptsBRISK.size() << " keypoints in " << 1000 * t / 1.0 << " ms\n";

    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::BRISK::create();
    cv::Mat descBRISK;
    t = (double)cv::getTickCount();
    descriptor->compute(imgGray, kptsBRISK, descBRISK);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK descriptor in " << 1000 * t / 1.0 << " ms" << endl;
    outfile << "BRISK descriptor in " << 1000 * t / 1.0 << " ms\n";

    // visualize results
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, kptsBRISK, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "BRISK Results";
    cv::namedWindow(windowName, 1);
    imshow(windowName, visImage);
    imwrite("../out/BRISK.jpg", visImage);

    // SIFT detector / descriptor
    detector = cv::xfeatures2d::SIFT::create();
    vector<cv::KeyPoint> kptsSIFT;

    t = (double)cv::getTickCount();
    detector->detect(imgGray, kptsSIFT);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "SIFT detector with n = " << kptsSIFT.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    outfile << "SIFT detector with n = " << kptsSIFT.size() << " keypoints in " << 1000 * t / 1.0 << " ms\n";

    descriptor = cv::xfeatures2d::SiftDescriptorExtractor::create();
    cv::Mat descSIFT;
    t = (double)cv::getTickCount();
    descriptor->compute(imgGray, kptsSIFT, descSIFT);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "SIFT descriptor in " << 1000 * t / 1.0 << " ms" << endl;
    outfile << "SIFT descriptor in " << 1000 * t / 1.0 << " ms\n";

    visImage = img.clone();
    cv::drawKeypoints(img, kptsSIFT, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    windowName = "SIFT Results";
    cv::namedWindow(windowName, 2);
    imshow(windowName, visImage);
    imwrite("../out/SIFT.jpg", visImage);

    // Clean up workspace
    outfile.close();
    cv::waitKey(0);

}

int main()
{
    descKeypoints1();
    return 0;
}