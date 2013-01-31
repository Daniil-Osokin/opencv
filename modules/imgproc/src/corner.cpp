/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#include "precomp.hpp"
#include <stdio.h>


namespace cv
{

static void
calcMinEigenVal( const Mat& _cov, Mat& _dst )
{
    int i, j;
    Size size = _cov.size();
#if CV_SSE
    volatile bool simd = checkHardwareSupport(CV_CPU_SSE);
#endif
    
    if( _cov.isContinuous() && _dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( i = 0; i < size.height; i++ )
    {
        const float* cov = (const float*)(_cov.data + _cov.step*i);
        float* dst = (float*)(_dst.data + _dst.step*i);
        j = 0;
    #if CV_SSE
        if( simd )
        {
            __m128 half = _mm_set1_ps(0.5f);
            for( ; j <= size.width - 5; j += 4 )
            {
                __m128 t0 = _mm_loadu_ps(cov + j*3); // a0 b0 c0 x
                __m128 t1 = _mm_loadu_ps(cov + j*3 + 3); // a1 b1 c1 x
                __m128 t2 = _mm_loadu_ps(cov + j*3 + 6); // a2 b2 c2 x
                __m128 t3 = _mm_loadu_ps(cov + j*3 + 9); // a3 b3 c3 x
                __m128 a, b, c, t;
                t = _mm_unpacklo_ps(t0, t1); // a0 a1 b0 b1
                c = _mm_unpackhi_ps(t0, t1); // c0 c1 x x
                b = _mm_unpacklo_ps(t2, t3); // a2 a3 b2 b3
                c = _mm_movelh_ps(c, _mm_unpackhi_ps(t2, t3)); // c0 c1 c2 c3
                a = _mm_movelh_ps(t, b);
                b = _mm_movehl_ps(b, t);
                a = _mm_mul_ps(a, half);
                c = _mm_mul_ps(c, half);
                t = _mm_sub_ps(a, c);
                t = _mm_add_ps(_mm_mul_ps(t, t), _mm_mul_ps(b,b));
                a = _mm_sub_ps(_mm_add_ps(a, c), _mm_sqrt_ps(t));
                _mm_storeu_ps(dst + j, a);
            }
        }
    #endif
        for( ; j < size.width; j++ )
        {
            float a = cov[j*3]*0.5f;
            float b = cov[j*3+1];
            float c = cov[j*3+2]*0.5f;
            dst[j] = (float)((a + c) - std::sqrt((a - c)*(a - c) + b*b));
        }
    }
}


static void
calcHarris( const Mat& _cov, Mat& _dst, double k )
{
    int i, j;
    Size size = _cov.size();
#if CV_SSE
    volatile bool simd = checkHardwareSupport(CV_CPU_SSE);
#endif
    
    if( _cov.isContinuous() && _dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( i = 0; i < size.height; i++ )
    {
        const float* cov = (const float*)(_cov.data + _cov.step*i);
        float* dst = (float*)(_dst.data + _dst.step*i);
        j = 0;

    #if CV_SSE
        if( simd )
        {
            __m128 k4 = _mm_set1_ps((float)k);
            for( ; j <= size.width - 5; j += 4 )
            {
                __m128 t0 = _mm_loadu_ps(cov + j*3); // a0 b0 c0 x
                __m128 t1 = _mm_loadu_ps(cov + j*3 + 3); // a1 b1 c1 x
                __m128 t2 = _mm_loadu_ps(cov + j*3 + 6); // a2 b2 c2 x
                __m128 t3 = _mm_loadu_ps(cov + j*3 + 9); // a3 b3 c3 x
                __m128 a, b, c, t;
                t = _mm_unpacklo_ps(t0, t1); // a0 a1 b0 b1
                c = _mm_unpackhi_ps(t0, t1); // c0 c1 x x
                b = _mm_unpacklo_ps(t2, t3); // a2 a3 b2 b3
                c = _mm_movelh_ps(c, _mm_unpackhi_ps(t2, t3)); // c0 c1 c2 c3
                a = _mm_movelh_ps(t, b);
                b = _mm_movehl_ps(b, t);
                t = _mm_add_ps(a, c);
                a = _mm_sub_ps(_mm_mul_ps(a, c), _mm_mul_ps(b, b));
                t = _mm_mul_ps(_mm_mul_ps(k4, t), t);
                a = _mm_sub_ps(a, t);
                _mm_storeu_ps(dst + j, a);
            }
        }
    #endif

        for( ; j < size.width; j++ )
        {
            float a = cov[j*3];
            float b = cov[j*3+1];
            float c = cov[j*3+2];
            dst[j] = (float)(a*c - b*b - k*(a + c)*(a + c));
        }
    }
}

void eigen2x2( const float* cov, float* dst, int n )
{
    for( int j = 0; j < n; j++ )
    {
        double a = cov[j*3];
        double b = cov[j*3+1];
        double c = cov[j*3+2];
        
        double u = (a + c)*0.5;
        double v = std::sqrt((a - c)*(a - c)*0.25 + b*b);
        double l1 = u + v;
        double l2 = u - v;
        
        double x = b;
        double y = l1 - a;
        double e = fabs(x);
        
        if( e + fabs(y) < 1e-4 )
        {
            y = b;
            x = l1 - c;
            e = fabs(x);
            if( e + fabs(y) < 1e-4 )
            {
                e = 1./(e + fabs(y) + FLT_EPSILON);
                x *= e, y *= e;
            }
        }
        
        double d = 1./std::sqrt(x*x + y*y + DBL_EPSILON);
        dst[6*j] = (float)l1;
        dst[6*j + 2] = (float)(x*d);
        dst[6*j + 3] = (float)(y*d);
        
        x = b;
        y = l2 - a;
        e = fabs(x);
        
        if( e + fabs(y) < 1e-4 )
        {
            y = b;
            x = l2 - c;
            e = fabs(x);
            if( e + fabs(y) < 1e-4 )
            {
                e = 1./(e + fabs(y) + FLT_EPSILON);
                x *= e, y *= e;
            }
        }
        
        d = 1./std::sqrt(x*x + y*y + DBL_EPSILON);
        dst[6*j + 1] = (float)l2;
        dst[6*j + 4] = (float)(x*d);
        dst[6*j + 5] = (float)(y*d);
    }
}

static void
calcEigenValsVecs( const Mat& _cov, Mat& _dst )
{
    Size size = _cov.size();
    if( _cov.isContinuous() && _dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( int i = 0; i < size.height; i++ )
    {
        const float* cov = (const float*)(_cov.data + _cov.step*i);
        float* dst = (float*)(_dst.data + _dst.step*i);

        eigen2x2(cov, dst, size.width);
    }
}


enum { MINEIGENVAL=0, HARRIS=1, EIGENVALSVECS=2 };

static void
cornerEigenValsVecs( const Mat& src, Mat& eigenv, int block_size,
                     int aperture_size, int op_type, double k=0.,
                     int borderType=BORDER_DEFAULT )
{
#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::cornerEigenValsVecs(src, eigenv, block_size, aperture_size, op_type, k, borderType))
        return;
#endif 
    int depth = src.depth();
    double scale = (double)(1 << ((aperture_size > 0 ? aperture_size : 3) - 1)) * block_size;
    if( aperture_size < 0 )
        scale *= 2.;
    if( depth == CV_8U )
        scale *= 255.;
    scale = 1./scale;

    CV_Assert( src.type() == CV_8UC1 || src.type() == CV_32FC1 );

    Mat Dx, Dy;
    if( aperture_size > 0 )
    {
        Sobel( src, Dx, CV_32F, 1, 0, aperture_size, scale, 0, borderType );
        Sobel( src, Dy, CV_32F, 0, 1, aperture_size, scale, 0, borderType );
    }
    else
    {
        Scharr( src, Dx, CV_32F, 1, 0, scale, 0, borderType );
        Scharr( src, Dy, CV_32F, 0, 1, scale, 0, borderType );
    }

    Size size = src.size();
    Mat cov( size, CV_32FC3 );
    int i, j;

    for( i = 0; i < size.height; i++ )
    {
        float* cov_data = (float*)(cov.data + i*cov.step);
        const float* dxdata = (const float*)(Dx.data + i*Dx.step);
        const float* dydata = (const float*)(Dy.data + i*Dy.step);

        for( j = 0; j < size.width; j++ )
        {
            float dx = dxdata[j];
            float dy = dydata[j];

            cov_data[j*3] = dx*dx;
            cov_data[j*3+1] = dx*dy;
            cov_data[j*3+2] = dy*dy;
        }
    }

    boxFilter(cov, cov, cov.depth(), Size(block_size, block_size),
        Point(-1,-1), false, borderType );

    if( op_type == MINEIGENVAL )
        calcMinEigenVal( cov, eigenv );
    else if( op_type == HARRIS )
        calcHarris( cov, eigenv, k );
    else if( op_type == EIGENVALSVECS )
        calcEigenValsVecs( cov, eigenv );
}

class CornerHarrisRunner
{
public:
    CornerHarrisRunner(Mat _src, Mat _dst, Mat _dstMaxVal, int _nStripes, int _blockSize, int _ksize, float _k, int _borderType)
    {
        src = _src;
        dst = _dst;
        dstMaxVal = _dstMaxVal;

        nStripes = _nStripes;

        blockSize = _blockSize;
        ksize = _ksize;
        k = _k;
        borderType = _borderType;
    }

    void operator () ( const BlockedRange& range ) const
    {
        int col0,col1;
        col0 = min(cvRound(range.begin() * src.cols / nStripes), src.cols);
        col1 = min(cvRound(range.end() * src.cols / nStripes), src.cols);
#if defined HAVE_TBB && defined ANDROID
            col0 = 8*cvRound((range.begin() * src.cols) / (8*nStripes));
            col1 = 8*cvRound((range.end() * src.cols) / (8*nStripes));
            if(range.end() == nStripes)
                col1 = src.cols;
#endif
        Mat srcStripe;
        if(nStripes != 1) {
            if(!range.begin()) {
                srcStripe = src.colRange(col0, col1+2*(blockSize/2+1));
            } else if(range.end() == 4) {
                if(blockSize !=5 )
                    srcStripe = src.colRange(col0-((col1-col0)%8), col1);//??
                else
                    srcStripe = src.colRange(col0-(blockSize/2+1), col1);
            } else {
                srcStripe = src.colRange(col0-(blockSize/2+1), col1+(blockSize/2+1));
            }
        }
        else
            srcStripe = src.colRange(col0, col1);
        Mat dstStripe;
        dstStripe.create(srcStripe.size(), CV_32F );
        cornerEigenValsVecs( srcStripe, dstStripe,  blockSize, ksize, 1, k, borderType);
        if(nStripes==1)
            minMaxLoc( dstStripe, 0, (double*)dstMaxVal.data);
        else {
            *((float*)dstMaxVal.data+range.begin()) = ((float*)dstStripe.data)[dstStripe.cols*dstStripe.rows-1];
        }
        Mat dstStripeEnd;
        if(nStripes==4) {
            if(!range.begin()) {
                dstStripeEnd = dstStripe.colRange(0, dstStripe.cols-2*(blockSize/2+1));
            } else if(range.end() == 4) {
                if(blockSize !=5 )
                    dstStripeEnd = dstStripe.colRange((2*(blockSize/2+1)), dstStripe.cols);
                else
                    dstStripeEnd = dstStripe.colRange(blockSize/2+1, dstStripe.cols);
            } else {
                dstStripeEnd = dstStripe.colRange(blockSize/2+1, dstStripe.cols-blockSize/2-1);
            }
        } else
            dstStripeEnd = dstStripe;
        dstStripeEnd.copyTo(dst.colRange(col0, col1));
        for( int i = 0; i < nStripes; i++)
        {
            *((float*)dstMaxVal.data) = std::max(*((float*)dstMaxVal.data), *((float*)dstMaxVal.data+i));
        }
    }

private:
    Mat src;
    Mat dst;
    Mat dstMaxVal;
    int nStripes;

    int blockSize;
    int ksize;
    float k;
    int borderType;
};

class CornerHarrisRunner2
{
public:
    CornerHarrisRunner2(Mat _src, vector<Point2f> _dstCorners[4], vector<float> _tmpCorners[4], int _nStripes, Mat _mask, int _blockSize, int _ksize, float _k, int _maxCorners, float _maxVal, float _qualityLevel, float _minDistance, int _borderType)
    {
        src = _src;
        nStripes = _nStripes;
        for( int i = 0; i < nStripes; i++) {
            dstCorners[i] = &_dstCorners[i];
            tmpCorners2[i] = &_tmpCorners[i];
        }

        mask = _mask;
        blockSize = _blockSize;
        ksize = _ksize;
        k = _k;
        maxCorners = _maxCorners;
        maxVal = _maxVal;
        qualityLevel = _qualityLevel;
        minDistance = _minDistance;
        borderType = _borderType;
    }

template<typename T> struct greaterThanPtr
{
    bool operator()(const T* a, const T* b) const { return *a > *b; }
};

    void operator () ( const BlockedRange& range ) const
    {
        int col0,col1;
        col0 = min(cvRound(range.begin() * src.cols / nStripes), src.cols);
        col1 = min(cvRound(range.end() * src.cols / nStripes), src.cols);
#if defined HAVE_TBB && defined ANDROID
            col0 = 8*cvRound((range.begin() * src.cols) / (8*nStripes));
            col1 = 8*cvRound((range.end() * src.cols) / (8*nStripes));
            if(range.end() == nStripes)
                col1 = src.cols;
#endif
        Mat srcStripe;
        if(nStripes != 1) {
            if(!range.begin()) {
                srcStripe = src.colRange(col0, col1+2*(blockSize/2+1));
            } else if(range.end() == 4) {
                if(blockSize !=5 )
                    srcStripe = src.colRange(col0-((col1-col0)%8), col1);//??
                else
                    srcStripe = src.colRange(col0-(blockSize/2+1), col1);
            } else {
                srcStripe = src.colRange(col0-(blockSize/2+1), col1+(blockSize/2+1));
            }
        }
        else
            srcStripe = src.colRange(col0, col1);
        Mat dstStripe;
        dstStripe.create(srcStripe.size(), CV_32F );

        threshold( srcStripe, srcStripe, maxVal*qualityLevel, 0, THRESH_TOZERO );
        dilate( srcStripe, dstStripe, Mat());

        vector<const float*> tmpCorners;
        for( int y = 3; y < srcStripe.rows - 3; y++ )
        {
            const float* eig_data = (const float*)srcStripe.ptr(y);
            const float* tmp_data = (const float*)dstStripe.ptr(y);
            const uchar* mask_data = mask.data ? mask.ptr(y) : 0;
            for( int x = 3; x < srcStripe.cols - 3; x++)
            {
                float val = eig_data[x];
                if( val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x])) {
                    tmpCorners.push_back(eig_data + x);
                }
            }
        }
        sort( tmpCorners, greaterThanPtr<float>() );
        vector<Point2f> corners;
        size_t i, j, total = tmpCorners.size(), ncorners = 0;

        Mat eig = srcStripe;
    if(minDistance >= 1)
    {    
        Mat image = eig;
         // Partition the image into larger grids
        int w = image.cols;
        int h = image.rows;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<Point2f> > grid(grid_width*grid_height);

        float minDistance1 = minDistance * minDistance;

        for( i = 0; i < total; i++ )
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.data);
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

	        bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width-1, x2);
            y2 = std::min(grid_height-1, y2);

            for( int yy = y1; yy <= y2; yy++ )
            {
                for( int xx = x1; xx <= x2; xx++ )
                {   
                    vector <Point2f> &m = grid[yy*grid_width + xx];

                    if( m.size() )
                    {
                        for(j = 0; j < m.size(); j++)
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if( dx*dx + dy*dy < minDistance1 )
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }                
                }
            }

            break_out:

            if(good)
            {
                // printf("%d: %d %d -> %d %d, %d, %d -- %d %d %d %d, %d %d, c=%d\n",
                //    i,x, y, x_cell, y_cell, (int)minDistance, cell_size,x1,y1,x2,y2, grid_width,grid_height,c);
                grid[y_cell*grid_width + x_cell].push_back(Point2f((float)x, (float)y));

                ++ncorners;

                if( maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    }
    else
    {
    }
        vector<float> tmpCorners1;
        for( i = 0; i < total; i++)
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.data);
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

            tmpCorners1.push_back(*(tmpCorners[i]));
            int ofsx = 2;
            if (range.begin()==1) 
                ofsx = 4; 
            if (range.begin()==2) 
                ofsx = 4; 
            else if (range.begin()==3) 
                ofsx = 6; 
            corners.push_back(Point2f((float)(x+col0-ofsx), (float)(y-2)));
            ++ncorners;
//            if( maxCorners > 0 && (int)ncorners == maxCorners )
//                break;
        }

          *(dstCorners[range.begin()]) = corners;
          *(tmpCorners2[range.begin()]) = tmpCorners1;
    }

private:
    Mat src;
    vector<Point2f>* dstCorners[4];
    vector<float>* tmpCorners2[4];
    int nStripes;

    Mat mask;
    int blockSize;
    int ksize;
    float k;
    int maxCorners;
    float maxVal;
    float qualityLevel;
    float minDistance;
    int borderType;
};
}

void cv::cornerMinEigenVal( InputArray _src, OutputArray _dst, int blockSize, int ksize, int borderType )
{
    Mat src = _src.getMat();
    _dst.create( src.size(), CV_32F );
    Mat dst = _dst.getMat();
    cornerEigenValsVecs( src, dst, blockSize, ksize, MINEIGENVAL, 0, borderType );
}
#if 0
void cv::cornerHarris( InputArray _src, OutputArray _dst, int blockSize, int ksize, double k, int borderType )
{
    Mat src = _src.getMat();
    _dst.create( src.size(), CV_32F );
    Mat dst = _dst.getMat();
    cornerEigenValsVecs( src, dst, blockSize, ksize, HARRIS, k, borderType );
}
#else

void cv::cornerHarris( InputArray _src, OutputArray _dst, int blockSize, int ksize, double k, int borderType )
{
    Mat src = _src.getMat();
    _dst.create( src.size(), CV_32F );
    Mat dst = _dst.getMat();
    int nStripes = 1;
#if defined HAVE_TBB && defined ANDROID
    nStripes = 4;
#endif
    Mat srcStripe;
    Mat dstStripe;
    Mat dstMaxVal;
    if (borderType == cv::BORDER_CONSTANT) {
        cornerEigenValsVecs( src, dst, blockSize, ksize, HARRIS, k, borderType );
    } else {
        copyMakeBorder(src, srcStripe, (blockSize/2)+1, (blockSize/2)+1, (blockSize/2)+1, (blockSize/2)+1, borderType, 0);
        dstMaxVal.create(1, nStripes, CV_32F);
        dstStripe.create( srcStripe.size(), CV_32F);
        parallel_for(BlockedRange(0, nStripes),
            CornerHarrisRunner(srcStripe, dstStripe, dstMaxVal, nStripes, blockSize, ksize, (float)k, borderType));
        dstStripe.colRange((blockSize/2)+1, dstStripe.cols-((blockSize/2)+1)).
            rowRange((blockSize/2)+1, dstStripe.rows-((blockSize/2)+1)).copyTo(dst);
        }
}
#endif

typedef std::pair<cv::Point2f, float> my_pair;

bool sort_pred(const my_pair& a, const my_pair& b)
{
    return a.second > b.second;
}

cv::Point2f firstElement( my_pair &p ) {
    return p.first;
}

void cv::cornerHarris( InputArray _src, OutputArray _dst, InputArray _mask, int blockSize, int ksize, double k, int* maxCorners , double* qualityLevel, double* minDistance, int borderType )
{
    Mat src = _src.getMat();
    Mat mask = _mask.getMat();
    double maxVal = 0;
    int nStripes = 1;
#if defined HAVE_TBB && defined ANDROID
    nStripes = 4;
#endif
    Mat srcStripe;
    Mat tmpStripe;
    Mat dstStripe;
    Mat dstMaxVal;
    if (borderType == cv::BORDER_CONSTANT) {
//        cornerEigenValsVecs( src, dst, blockSize, ksize, HARRIS, k, borderType );
    } else {
        copyMakeBorder(src, srcStripe, (blockSize/2)+1, (blockSize/2)+1, (blockSize/2)+1, (blockSize/2)+1, borderType, 0);
        dstMaxVal.create(nStripes, nStripes, CV_32F);
        for( int i = 0; i < nStripes; i++)
        {
            *((float*)dstMaxVal.data+i) = 0.0f;
        }
        tmpStripe.create( srcStripe.size(), CV_32F);
        dstStripe.create( srcStripe.size(), CV_32F);
        parallel_for(BlockedRange(0, nStripes),
            CornerHarrisRunner(srcStripe, tmpStripe, dstMaxVal, nStripes, blockSize, ksize, (float)k, borderType));
        if(nStripes==1)
            minMaxLoc( tmpStripe, 0, &maxVal);
        else
            maxVal = *((float*)dstMaxVal.data);
        vector<cv::Point2f> dstCorners[4];
        vector<float> tmpCorners[4];
        parallel_for(BlockedRange(0, nStripes),
            CornerHarrisRunner2(tmpStripe, dstCorners, tmpCorners, nStripes, mask, blockSize, ksize, (float)k, *maxCorners, (float)(maxVal), (float)*qualityLevel, (float)*minDistance, borderType));
        vector<cv::Point2f> dst;
        vector<float> myiter;
        for( int i = nStripes-1; i >= 0; i-- ) {
            dst.insert(dst.end(), dstCorners[i].begin(), dstCorners[i].end());
            myiter.insert(myiter.end(), tmpCorners[i].begin(), tmpCorners[i].end());
        }

        std::vector<my_pair> order(myiter.size());
        for (int n = 0; n != myiter.size(); n++)
            order[n] = std::make_pair(dst.at(n), myiter.at(n));
        std::sort(order.begin(), order.end(), sort_pred);

        vector<cv::Point2f> dstOut;
        std::transform( order.begin(), order.end(), std::back_inserter(dstOut), firstElement );
        Mat(dstOut).copyTo(_dst);
    }
}

void cv::cornerEigenValsAndVecs( InputArray _src, OutputArray _dst, int blockSize, int ksize, int borderType )
{
    Mat src = _src.getMat();
    Size dsz = _dst.size();
    int dtype = _dst.type();
    
    if( dsz.height != src.rows || dsz.width*CV_MAT_CN(dtype) != src.cols*6 || CV_MAT_DEPTH(dtype) != CV_32F )
        _dst.create( src.size(), CV_32FC(6) );
    Mat dst = _dst.getMat();
    cornerEigenValsVecs( src, dst, blockSize, ksize, EIGENVALSVECS, 0, borderType );
}


void cv::preCornerDetect( InputArray _src, OutputArray _dst, int ksize, int borderType )
{
    Mat Dx, Dy, D2x, D2y, Dxy, src = _src.getMat();

    CV_Assert( src.type() == CV_8UC1 || src.type() == CV_32FC1 );
    _dst.create( src.size(), CV_32F );
    Mat dst = _dst.getMat();
    
    Sobel( src, Dx, CV_32F, 1, 0, ksize, 1, 0, borderType );
    Sobel( src, Dy, CV_32F, 0, 1, ksize, 1, 0, borderType );
    Sobel( src, D2x, CV_32F, 2, 0, ksize, 1, 0, borderType );
    Sobel( src, D2y, CV_32F, 0, 2, ksize, 1, 0, borderType );
    Sobel( src, Dxy, CV_32F, 1, 1, ksize, 1, 0, borderType );

    double factor = 1 << (ksize - 1);
    if( src.depth() == CV_8U )
        factor *= 255;
    factor = 1./(factor * factor * factor);

    Size size = src.size();
    int i, j;
    for( i = 0; i < size.height; i++ )
    {
        float* dstdata = (float*)(dst.data + i*dst.step);
        const float* dxdata = (const float*)(Dx.data + i*Dx.step);
        const float* dydata = (const float*)(Dy.data + i*Dy.step);
        const float* d2xdata = (const float*)(D2x.data + i*D2x.step);
        const float* d2ydata = (const float*)(D2y.data + i*D2y.step);
        const float* dxydata = (const float*)(Dxy.data + i*Dxy.step);
        
        for( j = 0; j < size.width; j++ )
        {
            float dx = dxdata[j];
            float dy = dydata[j];
            dstdata[j] = (float)(factor*(dx*dx*d2ydata[j] + dy*dy*d2xdata[j] - 2*dx*dy*dxydata[j]));
        }
    }
}

CV_IMPL void
cvCornerMinEigenVal( const CvArr* srcarr, CvArr* dstarr,
                     int block_size, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size() == dst.size() && dst.type() == CV_32FC1 );
    cv::cornerMinEigenVal( src, dst, block_size, aperture_size, cv::BORDER_REPLICATE );
}

CV_IMPL void
cvCornerHarris( const CvArr* srcarr, CvArr* dstarr,
                int block_size, int aperture_size, double k )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size() == dst.size() && dst.type() == CV_32FC1 );
    cv::cornerHarris( src, dst, block_size, aperture_size, k, cv::BORDER_REPLICATE );
}

CV_IMPL void
cvCornerEigenValsAndVecs( const void* srcarr, void* dstarr,
                          int block_size, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.rows == dst.rows && src.cols*6 == dst.cols*dst.channels() && dst.depth() == CV_32F );
    cv::cornerEigenValsAndVecs( src, dst, block_size, aperture_size, cv::BORDER_REPLICATE );
}


CV_IMPL void
cvPreCornerDetect( const void* srcarr, void* dstarr, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size() == dst.size() && dst.type() == CV_32FC1 );
    cv::preCornerDetect( src, dst, aperture_size, cv::BORDER_REPLICATE );
}

/* End of file */
