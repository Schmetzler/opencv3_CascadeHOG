/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace cv
{
    using std::vector;
#define CCC_CASCADE_PARAMS "cascadeParams"
#define CCC_STAGE_TYPE     "stageType"
#define CCC_FEATURE_TYPE   "featureType"
#define CCC_HEIGHT         "height"
#define CCC_WIDTH          "width"

#define CCC_STAGE_NUM    "stageNum"
#define CCC_STAGES       "stages"
#define CCC_STAGE_PARAMS "stageParams"

#define CCC_BOOST            "BOOST"
#define CCC_MAX_DEPTH        "maxDepth"
#define CCC_WEAK_COUNT       "maxWeakCount"
#define CCC_STAGE_THRESHOLD  "stageThreshold"
#define CCC_WEAK_CLASSIFIERS "weakClassifiers"
#define CCC_INTERNAL_NODES   "internalNodes"
#define CCC_LEAF_VALUES      "leafValues"

#define CCC_FEATURES       "features"
#define CCC_FEATURE_PARAMS "featureParams"
#define CCC_MAX_CAT_COUNT  "maxCatCount"

#define CCV_SUM_PTRS( p0, p1, p2, p3, sum, rect, step )                    \
    /* (x, y) */                                                          \
    (p0) = sum + (rect).x + (step) * (rect).y,                            \
    /* (x + w, y) */                                                      \
    (p1) = sum + (rect).x + (rect).width + (step) * (rect).y,             \
    /* (x + w, y) */                                                      \
    (p2) = sum + (rect).x + (step) * ((rect).y + (rect).height),          \
    /* (x + w, y + h) */                                                  \
    (p3) = sum + (rect).x + (rect).width + (step) * ((rect).y + (rect).height)

//#define CCV_TILTED_PTRS( p0, p1, p2, p3, tilted, rect, step )                        \
//    /* (x, y) */                                                                    \
//    (p0) = tilted + (rect).x + (step) * (rect).y,                                   \
//    /* (x - h, y + h) */                                                            \
//    (p1) = tilted + (rect).x - (rect).height + (step) * ((rect).y + (rect).height), \
//    /* (x + w, y + w) */                                                            \
//    (p2) = tilted + (rect).x + (rect).width + (step) * ((rect).y + (rect).width),   \
//    /* (x + w - h, y + w + h) */                                                    \
//    (p3) = tilted + (rect).x + (rect).width - (rect).height                         \
//           + (step) * ((rect).y + (rect).width + (rect).height)

#define CCALC_SUM_(p0, p1, p2, p3, offset) \
    ((p0)[offset] - (p1)[offset] - (p2)[offset] + (p3)[offset])

#define CCALC_SUM(rect,offset) CCALC_SUM_((rect)[0], (rect)[1], (rect)[2], (rect)[3], offset)

//#define CCC_HAAR   "HAAR"
//#define CCC_RECTS  "rects"
//#define CCC_TILTED "tilted"
//
//#define CCC_LBP  "LBP"
#define CCC_RECT "rect"

#define CCC_HOG  "HOG"

class HOGEvaluator
{
public:
    static const int HOG = 2;
    struct Feature
    {
        Feature();
        float calc( int offset ) const;
        void updatePtrs( const vector<Mat>& _hist, const Mat &_normSum );
        bool read( const FileNode& node );

        enum { CELL_NUM = 4, BIN_NUM = 9 };

        Rect rect[CELL_NUM];
        int featComponent; //component index from 0 to 35
        const float* pF[4]; //for feature calculation
        const float* pN[4]; //for normalization calculation
    };
    HOGEvaluator();
    virtual ~HOGEvaluator();
    virtual bool read( const FileNode& node );
    virtual Ptr<HOGEvaluator> clone() const;
    virtual int getFeatureType() const { return HOGEvaluator::HOG; }
    virtual bool setImage( const Mat& image, Size winSize );
    virtual bool setWindow( Point pt );
    double operator()(int featureIdx) const
    {
        return featuresPtr[featureIdx].calc(offset);
    }
    virtual double calcOrd( int featureIdx ) const
    {
        return (*this)(featureIdx);
    }

private:
    virtual void integralHistogram( const Mat& srcImage, vector<Mat> &histogram, Mat &norm, int nbins ) const;

    Size origWinSize;
    Ptr<vector<Feature> > features;
    Feature* featuresPtr;
    vector<Mat> hist;
    Mat normSum;
    int offset;
};

class CV_EXPORTS_W HOGCascadeClassifier
{
public:    
    CV_WRAP HOGCascadeClassifier();
    CV_WRAP HOGCascadeClassifier(const std::string& filename);
    virtual ~HOGCascadeClassifier();
    CV_WRAP virtual bool empty() const;
    CV_WRAP bool load(const std::string& filename);
    virtual bool read( const FileNode& node );
    CV_WRAP void detectMultiScale(const Mat& image,
                    CV_OUT std::vector<Rect>& objects,
                    double scaleFactor = 1.1,
                    int minNeighbors = 3, int flags = 0,
                    Size minSize = Size(),
                    Size maxSize = Size());
                    
    CV_WRAP void detectMultiScale( const Mat& image, vector<Rect>& objects,
                    vector<int>& rejectLevels,
                    vector<double>& levelWeights,
                    double scaleFactor=1.1,
                    int minNeighbors=3, int flags=0,
                    Size minSize=Size(),
                    Size maxSize=Size(),
                    bool outputRejectLevels=false );
                    
    class CV_EXPORTS MaskGenerator
    {
    public:
        virtual ~MaskGenerator() {}
        virtual cv::Mat generateMask(const cv::Mat& src)=0;
        virtual void initializeMask(const cv::Mat& /*src*/) {};
    };
    void setMaskGenerator(Ptr<MaskGenerator> maskGenerator);
    Ptr<MaskGenerator> getMaskGenerator();
                    
protected:
    Ptr<MaskGenerator> maskGenerator;
    virtual int runAt( Ptr<HOGEvaluator>& feval, Point pt, double& weight );
    virtual bool detectSingleScale( const Mat& image, int stripCount, Size processingRectSize,
                                    int stripSize, int yStep, double factor, vector<Rect>& candidates,
                                    vector<int>& rejectLevels, vector<double>& levelWeights, bool outputRejectLevels=false);

    friend class HOGCascadeClassifierInvoker;

    friend int HOGpredictOrdered( HOGCascadeClassifier& cascade, Ptr<HOGEvaluator> &featureEvaluator, double& weight);
    friend int HOGpredictOrderedStump( HOGCascadeClassifier& cascade, Ptr<HOGEvaluator> &featureEvaluator, double& weight);

    class Data
    {
        public:
        struct CV_EXPORTS DTreeNode
        {
            int featureIdx;
            float threshold; // for ordered features only
            int left;
            int right;
        };

        struct CV_EXPORTS DTree
        {
            int nodeCount;
        };

        struct CV_EXPORTS Stage
        {
            int first;
            int ntrees;
            float threshold;
        };

        bool read(const FileNode &node);

        bool isStumpBased;

        int stageType;
        int featureType;
        int ncategories;
        Size origWinSize;

        vector<Stage> stages;
        vector<DTree> classifiers;
        vector<DTreeNode> nodes;
        vector<float> leaves;
        vector<int> subsets;
    };
    
    Data data;
    Ptr<HOGEvaluator> featureEvaluator;
};

inline HOGEvaluator::Feature :: Feature()
{
    rect[0] = rect[1] = rect[2] = rect[3] = Rect();
    pF[0] = pF[1] = pF[2] = pF[3] = 0;
    pN[0] = pN[1] = pN[2] = pN[3] = 0;
    featComponent = 0;
}

inline float HOGEvaluator::Feature :: calc( int _offset ) const
{
    float res = CCALC_SUM(pF, _offset);
    float normFactor = CCALC_SUM(pN, _offset);
    res = (res > 0.001f) ? (res / ( normFactor + 0.001f) ) : 0.f;
    return res;
}

inline void HOGEvaluator::Feature :: updatePtrs( const vector<Mat> &_hist, const Mat &_normSum )
{
    int binIdx = featComponent % BIN_NUM;
    int cellIdx = featComponent / BIN_NUM;
    Rect normRect = Rect( rect[0].x, rect[0].y, 2*rect[0].width, 2*rect[0].height );

    const float* featBuf = (const float*)_hist[binIdx].data;
    size_t featStep = _hist[0].step / sizeof(featBuf[0]);

    const float* normBuf = (const float*)_normSum.data;
    size_t normStep = _normSum.step / sizeof(normBuf[0]);

    CCV_SUM_PTRS( pF[0], pF[1], pF[2], pF[3], featBuf, rect[cellIdx], featStep );
    CCV_SUM_PTRS( pN[0], pN[1], pN[2], pN[3], normBuf, normRect, normStep );
}

inline int HOGpredictOrdered( HOGCascadeClassifier& cascade, Ptr<HOGEvaluator> &_featureEvaluator, double& sum )
{
    int nstages = (int)cascade.data.stages.size();
    int nodeOfs = 0, leafOfs = 0;
    HOGEvaluator& featureEvaluator = (HOGEvaluator&)*_featureEvaluator;
    float* cascadeLeaves = &cascade.data.leaves[0];
    HOGCascadeClassifier::Data::DTreeNode* cascadeNodes = &cascade.data.nodes[0];
    HOGCascadeClassifier::Data::DTree* cascadeWeaks = &cascade.data.classifiers[0];
    HOGCascadeClassifier::Data::Stage* cascadeStages = &cascade.data.stages[0];

    for( int si = 0; si < nstages; si++ )
    {
        HOGCascadeClassifier::Data::Stage& stage = cascadeStages[si];
        int wi, ntrees = stage.ntrees;
        sum = 0;

        for( wi = 0; wi < ntrees; wi++ )
        {
            HOGCascadeClassifier::Data::DTree& weak = cascadeWeaks[stage.first + wi];
            int idx = 0, root = nodeOfs;

            do
            {
                HOGCascadeClassifier::Data::DTreeNode& node = cascadeNodes[root + idx];
                double val = featureEvaluator(node.featureIdx);
                idx = val < node.threshold ? node.left : node.right;
            }
            while( idx > 0 );
            sum += cascadeLeaves[leafOfs - idx];
            nodeOfs += weak.nodeCount;
            leafOfs += weak.nodeCount + 1;
        }
        if( sum < stage.threshold )
            return -si;
    }
    return 1;
}

inline int HOGpredictOrderedStump( HOGCascadeClassifier& cascade, Ptr<HOGEvaluator> &_featureEvaluator, double& sum )
{
    int nodeOfs = 0, leafOfs = 0;
    HOGEvaluator& featureEvaluator = (HOGEvaluator&)*_featureEvaluator;
    float* cascadeLeaves = &cascade.data.leaves[0];
    HOGCascadeClassifier::Data::DTreeNode* cascadeNodes = &cascade.data.nodes[0];
    HOGCascadeClassifier::Data::Stage* cascadeStages = &cascade.data.stages[0];

    int nstages = (int)cascade.data.stages.size();
    for( int stageIdx = 0; stageIdx < nstages; stageIdx++ )
    {
        HOGCascadeClassifier::Data::Stage& stage = cascadeStages[stageIdx];
        sum = 0.0;

        int ntrees = stage.ntrees;
        for( int i = 0; i < ntrees; i++, nodeOfs++, leafOfs+= 2 )
        {
            HOGCascadeClassifier::Data::DTreeNode& node = cascadeNodes[nodeOfs];
            double value = featureEvaluator(node.featureIdx);
            sum += cascadeLeaves[ value < node.threshold ? leafOfs : leafOfs + 1 ];
        }

        if( sum < stage.threshold )
            return -stageIdx;
    }

    return 1;
}

}
