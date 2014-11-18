#include <iostream>
#include <fstream>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;

namespace
{
const String keys =
        "{help h      | | print this message       }"
        "{@cascade    | | path to cascade xml file }"
        "{@imgs-list  | | path to test images list }";

struct TestSample
{
    Mat img;
    vector<Rect> goldObjs;
};

class ITestSamplesParser
{
public:
    TestSample getSample(size_t id) const { return testSamples_[id]; }
    size_t getSamplesSize() const { return testSamples_.size(); }

protected:
    ITestSamplesParser() {}
    vector<TestSample> testSamples_;
};

// For http://vasc.ri.cmu.edu/idb/html/face/frontal_images/ dataset
class CmuFdTestSamplesParser: public ITestSamplesParser
{
public:
    CmuFdTestSamplesParser(ifstream& testImagesList, const string& dirPref = "")
    {
        CV_Assert(testImagesList.is_open());
        Objs objs;
        string filename;
        vector<float> coords(12);

        readLine(testImagesList, filename, coords);
        while (!filename.empty())
        {
            objs[filename].push_back(coords);
            filename = "";
            readLine(testImagesList, filename, coords);
        }

        for (Objs::const_iterator it = objs.begin(); it != objs.end(); it++)
        {
            filename = dirPref + it->first;
            TestSample sample;
            sample.img = imread(filename, IMREAD_GRAYSCALE);
            CV_Assert(!sample.img.empty());
            for (size_t i = 0; i < it->second.size(); i++)
            {
                coords = it->second[i];

                float minX = min(min(coords[0], coords[2]),
                        min(min(coords[4], coords[6]), min(coords[8], coords[10])));
                float minY = min(min(coords[1], coords[3]),
                        min(min(coords[5], coords[7]), min(coords[9], coords[11])));

                float maxX = max(max(coords[0], coords[2]),
                        max(max(coords[4], coords[6]), max(coords[8], coords[10])));
                float maxY = max(max(coords[1], coords[3]),
                        max(max(coords[5], coords[7]), max(coords[9], coords[11])));
                sample.goldObjs.push_back(Rect(Point(minX, minY), Point(maxX, maxY)));
            }
            testSamples_.push_back(sample);
        }
    }

private:
    typedef map<string, vector<vector<float> > > Objs;
    void readLine(ifstream& is, string& filename,
                  vector<float>& coords) const
    {
        is >> filename;
        for (size_t i = 0; i < coords.size(); i++)
        {
            is >> coords[i];
        }
    }
};
} // namespace

int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    String cascadePath = parser.get<String>(0);
    String testImagesPath = parser.get<String>(1);
    if (cascadePath.empty() || testImagesPath.empty())
    {
        cout << "Error: cascade or imgs-list parameter is omitted" << endl;
        parser.printMessage();
        return EXIT_FAILURE;
    }

    cv::Ptr<cv::CascadeClassifier> cascade = makePtr<cv::CascadeClassifier>(cascadePath);
    if (cascade->empty())
    {
        cout << "Cannot create classifier from path: " << cascadePath << endl;
        return EXIT_FAILURE;
    }

    ifstream testImagesList(testImagesPath.c_str());
    if (!testImagesList.is_open())
    {
        cout << "Cannot open test images file from path: " << testImagesPath << endl;
        return EXIT_FAILURE;
    }

    CmuFdTestSamplesParser p(testImagesList, testImagesPath.substr(0, testImagesPath.find_last_of('/') + 1));
    const size_t size = p.getSamplesSize();
    size_t goldNum = 0;
    size_t testNum = 0;
    size_t matchedNum = 0;
    for (size_t id = 0; id < size; id++)
    {
        const TestSample& sample = p.getSample(id);
        vector<Rect> testObjs;
        cascade->detectMultiScale(sample.img, testObjs);
        testNum += testObjs.size();
        vector<Rect> goldObjs = sample.goldObjs;
        vector<Rect> matchedTest;
        for (size_t i = 0; i < goldObjs.size(); i++)
        {
            goldNum++;
            for (size_t j = 0; j < testObjs.size(); j++)
            {
                if ((goldObjs[i] & testObjs[j]).area() > goldObjs[i].area()*0.8
                        && (16*goldObjs[i].area() >= testObjs[j].area()))
                {
                    matchedTest.push_back(testObjs[j]);
                    matchedNum++;
                    break;
                }
            }
        }


        /*Mat res = sample.img.clone();
        for (size_t j = 0; j < sample.goldObjs.size(); j++)
        {
            rectangle(res, sample.goldObjs[j], Scalar(255));
        }
        for (size_t j = 0; j < testObjs.size(); j++)
        {
            rectangle(res, testObjs[j], Scalar(128));
        }
        for (size_t j = 0; j < matchedTest.size(); j++)
        {
            rectangle(res, matchedTest[j], Scalar(0), 2);
        }
        imshow("res", res);
        waitKey();*/
    }
    cout << "Recall: " << (float)matchedNum / goldNum << endl;
    cout << "Precision: " << (float)matchedNum / testNum << endl;
    cout << "Gold obj: " << goldNum << ", Test obj: " << testNum
         << ", matched: " << matchedNum << endl;

    return EXIT_SUCCESS;
}
