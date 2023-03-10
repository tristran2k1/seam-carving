#include <stdio.h>
#include <stdint.h>
#include "../kernel.cuh"
#include "../utils.cuh"

__global__ void blurImg_kernel_v1(int *inPixels, int width, int height, int *outPixels){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // apply convolution on each pixel
    if (ix < width && iy < height)
    {
        int idx_1d = iy * width + ix;
        int ele = 0;

        for (int dy = -BLUR_KERNEL_SIZE / 2; dy <= BLUR_KERNEL_SIZE / 2; dy++)
        {
            for (int dx = -BLUR_KERNEL_SIZE / 2; dx <= BLUR_KERNEL_SIZE / 2; dx++)
            {

                int conv_x = max(min(ix + dx, width - 1), 0);
                int conv_y = max(min(iy + dy, height - 1), 0);
                int filter_x = dx + BLUR_KERNEL_SIZE / 2;
                int filter_y = dy + BLUR_KERNEL_SIZE / 2;
                float ele_conv = DEVICE_BLUR_KERNEL[filter_y * BLUR_KERNEL_SIZE + filter_x];

                ele += int(inPixels[conv_y * width + conv_x] * ele_conv);
            }
        }

        outPixels[idx_1d] = ele;
    }
}

__global__ void calcEnergyMap_kernel_v1(int *inPixels, int width, int height, int *outPixels)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // apply convolution on each pixel
    if (ix < width && iy < height)
    {
        int idx_1d = iy * width + ix;
        int ele_x = 0, ele_y = 0;

        for (int dy = -SOBEL_KERNEL_SIZE / 2; dy <= SOBEL_KERNEL_SIZE / 2; dy++)
        {
            for (int dx = -SOBEL_KERNEL_SIZE / 2; dx <= SOBEL_KERNEL_SIZE / 2; dx++)
            {

                int conv_x = max(min(ix + dx, width - 1), 0);
                int conv_y = max(min(iy + dy, height - 1), 0);
                int filter_x = dx + SOBEL_KERNEL_SIZE / 2;
                int filter_y = dy + SOBEL_KERNEL_SIZE / 2;

                int ele_gx = DEVICE_SOBELX_KERNEL[filter_y * SOBEL_KERNEL_SIZE + filter_x];
                int ele_gy = DEVICE_SOBELY_KERNEL[filter_y * SOBEL_KERNEL_SIZE + filter_x];

                ele_x += int(inPixels[conv_y * width + conv_x] * ele_gx);
                ele_y += int(inPixels[conv_y * width + conv_x] * ele_gy);
            }
        }

        outPixels[idx_1d] = (int)sqrt(float(ele_x * ele_x + ele_y * ele_y));
    }
}

void removeSingleSeam(uchar3 *inPixels, int width, int height, int seam_order, uchar3 *outPixels, int blocksize)
{
    dim3 blockSize(blocksize, blocksize);
    // 0. Preparation
    /* Declare variables */
    uchar3 *src = inPixels;

    // Variables
    uchar3 *d_src;
    int *d_src_gray, *d_src_blur, *d_energies;

    int *min_path = (int *)malloc(height * sizeof(int));
    int *min_row0 = (int *)malloc(width * sizeof(int));
    int *min_array = (int *)malloc(width * height * sizeof(int));

    size_t pixelsSize_3channels = width * height * sizeof(uchar3);
    size_t pixelsSize_1channels = width * height * sizeof(int);

    CHECK(cudaMalloc(&d_src, pixelsSize_3channels));
    CHECK(cudaMemcpy(d_src, src, pixelsSize_3channels, cudaMemcpyHostToDevice));

    // 1. Convert img to gray
    CHECK(cudaMalloc(&d_src_gray, pixelsSize_1channels));
    dim3 gridSize_gray((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

    convertRgb2Gray_kernel<<<gridSize_gray, blockSize>>>(d_src, width, height, d_src_gray);

    // 2. Blur gray img
    CHECK(cudaMalloc(&d_src_blur, pixelsSize_1channels));
    dim3 gridSize_blur((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

    blurImg_kernel_v1<<<gridSize_blur, blockSize>>>(d_src_gray, width, height, d_src_blur);

    // 3. Calc Energies
    CHECK(cudaMalloc(&d_energies, pixelsSize_1channels));

    calcEnergyMap_kernel_v1<<<gridSize_blur, blockSize>>>(d_src_blur, width, height, d_energies);

    int *energy = (int *)malloc(pixelsSize_1channels);
    CHECK(cudaMemcpy(energy, d_energies, pixelsSize_1channels, cudaMemcpyDeviceToHost));

    // 4. Find min cost each row iteratively (dp - iterative)
    int min_seam_idx = -1;
    findSeam(energy, width, height, min_seam_idx, min_array);

    if (min_seam_idx == -1)
    {
        printf("cannot find min seam at %d\n", seam_order);
    }

    // 5. Calculate min col idx on each row
    min_path[0] = min_seam_idx;

    for (int i = 1; i < height; ++i)
    {
        min_path[i] = min_array[(i - 1) * width + min_path[i - 1]];
    }

    // 6. Remove scene
    uchar3 *d_output;
    int *d_min_path;

    CHECK(cudaMalloc(&d_output, (width - 1) * height * sizeof(uchar3)));
    CHECK(cudaMalloc(&d_min_path, height * sizeof(int)));
    CHECK(cudaMemcpy(d_min_path, min_path, height * sizeof(int), cudaMemcpyHostToDevice));

    removeSeam<<<gridSize_blur, blockSize>>>(d_src, width, height, d_min_path, d_output);
    CHECK(cudaMemcpy(outPixels, d_output, (width - 1) * height * sizeof(uchar3), cudaMemcpyDeviceToHost));

    // Clean memory
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_energies));
    CHECK(cudaFree(d_src_blur));
    CHECK(cudaFree(d_src_gray));

    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_min_path));

    free(min_array);
    free(min_path);
    free(min_row0);
}

void applySeamCarving(uchar3 *inPixels, int width, int height, int nSeams, uchar3 *&outPixels, int blocksize)
{
    float *gx, *gy;
    createSobelFilters(gx, gy);
    float *gaussFilter;
    createGaussianFilter(gaussFilter);

    // copy data
    CHECK(cudaMemcpyToSymbol(DEVICE_BLUR_KERNEL, gaussFilter, BLUR_KERNEL_SIZE * BLUR_KERNEL_SIZE * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(DEVICE_SOBELX_KERNEL, gx, SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(DEVICE_SOBELY_KERNEL, gy, SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE * sizeof(float)));

    int step_width = 0;
    uchar3 *src = inPixels;

    for (int seam_order = 0; seam_order < nSeams; seam_order++)
    {
        step_width = width - seam_order;
        if (seam_order == 0)
            outPixels = (uchar3 *)malloc((step_width - 1) * height * sizeof(uchar3));
        else
            outPixels = (uchar3 *)realloc(outPixels, (step_width - 1) * height * sizeof(uchar3));

        removeSingleSeam(src, step_width, height, seam_order + 1, outPixels, blocksize);

        src = outPixels;
    }
}

int main(int argc, char ** argv) {
    printDeviceInfo();
    int blocksize = 32;
    GpuTimer timer;

    int width, height;
    uchar3 *inPixels = NULL;
    uchar3 *outPixels = NULL;
    int nSeams = atoi(argv[3]);
    readPnm(argv[1], width, height, inPixels);
    char * outFileNameBase = strtok(argv[2], ".");

    timer.Start();
    applySeamCarving(inPixels, width, height, nSeams, outPixels, blocksize);
    timer.Stop();

    float kernelTime = timer.Elapsed();
    printf("Version GPU 1, seams: %d, time: %f ms\n", nSeams, kernelTime);

    writePnm(outPixels, width - nSeams, height,  concatStr(outFileNameBase, "_dv1.pnm"));
}