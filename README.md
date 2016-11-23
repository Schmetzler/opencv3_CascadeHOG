# OpenCV3 CascadeHOG
The OpenCV2 HOG cascade ported for OpenCV3

## Introduction

This contains the opencv2.x implementation for the HOGEvaluator. I just copied the needed sources and made extra classes out of it. With this its possible to use HOG trained cascade models in OpenCV3.x.

You can build a library out of it or add it directly into your code.

I couldn't add it properly to the Classifier and Evaluator Framework so it contains extra classes:

* cv::HOGCascadeClassifier (Its basically the same as CascadeClassifier, but just for HOGEvaluators)
* cv::HOGEvaluator (The Evaluator it does not inherit from FeatureEvaluator but it could)

## Usage

To use it somehow like the original CascadeClassifier I use it like this:

```c++
cv::CascadeClassifier classifier;
cv::HOGCascadeClassifier hogclassifier;

// must check for HOG first, because OpenCV3 will throw an error otherwise
if(!hogclassifier.load("somefile.xml")
    if(!classifier.load("somefile.xml")
        throw exception("Could not read file");

...

if(!hogclassifier.empty())
    hogclassifier.detectMultiScale(...);
else
    classifier.detectMultiScale(...);
```

## Build instructions

You may build a library out of it with something like (on linux):

```
g++ -std=c++11 -fPIC -c -o hogcascade.o hogcascade.cpp
g++ -shared -Wl,--no-undefined -o hogcascade.so hogcascade.o -lopencv_core -lopencv_objdetect -lopencv_imgproc
```

## Licence

This is under the same Licence as OpenCV (well because it is from OpenCV), see LICENCE for more info.

