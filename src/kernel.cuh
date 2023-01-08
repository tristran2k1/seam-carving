#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <stdint.h>

#define SOBEL_KERNEL_SIZE 3
#define BLUR_KERNEL_SIZE 3

// CUDA AREA
__constant__ float DEVICE_BLUR_KERNEL[BLUR_KERNEL_SIZE * BLUR_KERNEL_SIZE];
__constant__ float DEVICE_SOBELX_KERNEL[SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE];
__constant__ float DEVICE_SOBELY_KERNEL[SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE];

void createGaussianFilter(float *&filter)
{
    /*
        Create a 3x3 Gaussian filter
        Expected output with KERNEL_SIZE = 3
        [1/16, 2/16, 1/16]
        [2/16, 4/16, 2/16]
        [1/16, 2/16, 1/16]
    */

    int filterWidth = BLUR_KERNEL_SIZE;
    filter = (float *)malloc(filterWidth * filterWidth * sizeof(float));
    for (int filterR = 0; filterR < filterWidth; filterR++)
    {
        for (int filterC = 0; filterC < filterWidth; filterC++)
        {
            filter[filterR * filterWidth + filterC] = 1.0f / (filterWidth * filterWidth);
        }
    }
}

void createSobelFilters(float *&gx, float *&gy)
{
    /*
    gx =    [-1,-2,-1] 
            [ 0, 0, 0] 
            [ 1, 2, 1] 

    gy =    [-1, 0, 1] 
            [-2, 0, 2] 
            [-1, 0, 1] 
            
    
    G = |gx| + |gy|

    */
    gx = (float *)malloc(SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE * sizeof(float));
    gy = (float *)malloc(SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE * sizeof(float));

    gx[3 * 0 + 0] = -1, gx[3 * 0 + 1] = 0, gx[3 * 0 + 2] = 1;
    gx[3 * 1 + 0] = -2, gx[3 * 1 + 1] = 0, gx[3 * 1 + 2] = 2;
    gx[3 * 2 + 0] = -1, gx[3 * 2 + 1] = 0, gx[3 * 2 + 2] = 1;

    gy[3 * 0 + 0] = -1, gy[3 * 0 + 1] = -2, gy[3 * 0 + 2] = -1;
    gy[3 * 1 + 0] = 0, gy[3 * 1 + 1] = 0, gy[3 * 1 + 2] = 0;
    gy[3 * 2 + 0] = 1, gy[3 * 2 + 1] = 2, gy[3 * 2 + 2] = 1;
}

int getMinCost(int *energy, int width, int height, int x, int y)
{
    int minEnergy = INT_MAX;
    int minIdx = -1;
    int neighbor[3] = {-1, 0, 1};
    for (int i = 0; i < 3; i++)
    {
        int x_ = min(max(0, x + neighbor[i]), width - 1);
        int y_ = y + 1;

        int cost = energy[width * y_ + x_] + energy[width * y + x];
        if (cost < minEnergy)
        {
            minEnergy = cost;
            minIdx = x_;
        }
    }

    energy[width * y + x] = minEnergy;
    return minIdx;
}

void findSeam(int *energy, int width, int height, int &seamIdx, int *path)
{
    // 1. dp
    for (int y = height - 2; y >= 0; y--)
    {
        for (int x = 0; x < width; x++)
        {
            int minIdx = getMinCost(energy, width, height, x, y);
            path[width * y + x] = minIdx;
        }
    }

    // 2. Choose min seam
    int minSeamIdx = -1;
    int minSeamCost = INT_MAX;
    for (int i = 0; i < width; i++)
    {
        if (energy[i] < minSeamCost)
        {
            minSeamCost = energy[i];
            minSeamIdx = i;
        }
    }

    seamIdx = minSeamIdx;
    // printf("CPU min val: %d\n", minSeamCost);
}

// Convert image to grayscale kernel
__global__ void convertRgb2Gray_kernel(uchar3 *inPixels, int width, int height, int *outPixels)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < width && iy < height)
    {
        int idx = iy * width + ix;
        int r = inPixels[idx].x;
        int g = inPixels[idx].y;
        int b = inPixels[idx].z;
        outPixels[idx] = int(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Remove min seam
__global__ void removeSeam(uchar3 *input, int width, int height, int *path, uchar3 *output)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    if (iy < height && ix < width)
    {
        int index_seam = path[iy];

        if (ix < index_seam)
        {
            output[iy * (width - 1) + ix].x = input[iy * width + ix].x;
            output[iy * (width - 1) + ix].y = input[iy * width + ix].y;
            output[iy * (width - 1) + ix].z = input[iy * width + ix].z;
        }
        else if (ix > index_seam && ix < width)
        {
            output[iy * (width - 1) + ix - 1].x = input[iy * width + ix].x;
            output[iy * (width - 1) + ix - 1].y = input[iy * width + ix].y;
            output[iy * (width - 1) + ix - 1].z = input[iy * width + ix].z;
        }
    }
}

#endif /* KERNEL_H */
