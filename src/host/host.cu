#include <stdio.h>
#include <stdint.h>
#include "../kernel.cuh"
#include "../utils.cuh"

void convertRgb2Gray(uchar3 *inPixels, int width, int height, int *out)
{
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int i = r * width + c;
            int r = inPixels[i].x;
            int g = inPixels[i].y;
            int b = inPixels[i].z;
            out[i] = int(0.299f * r + 0.587f * g + 0.114f * b);
            ;
        }
    }
}

void removeSeam(uchar3 *inPixels, int width, int height, int seamIdx, int *path, uchar3 *outPixels)
{
    int delimIdx = seamIdx;
    copyRow(inPixels, width, height, delimIdx, 0, outPixels);

    for (int i = 1; i < height; i++)
    {
        delimIdx = path[(i - 1) * width + delimIdx];
        copyRow(inPixels, width, height, delimIdx, i, outPixels);
    }
}

void calConvolution(int *grayPixels, int width, int height, float *filter, int filterWidth, int *outPixels)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx_1d = y * width + x;
            int ele = 0;

            for (int dy = -filterWidth / 2; dy <= filterWidth / 2; dy++)
            {
                for (int dx = -filterWidth / 2; dx <= filterWidth / 2; dx++)
                {
                    int conv_x = max(min(x + dx, width - 1), 0);
                    int conv_y = max(min(y + dy, height - 1), 0);

                    int filter_x = dx + filterWidth / 2;
                    int filter_y = dy + filterWidth / 2;
                    float ele_conv = filter[filter_y * filterWidth + filter_x];

                    ele += int(grayPixels[conv_y * width + conv_x] * ele_conv);
                }
            }

            outPixels[idx_1d] = (int)ele;
        }
    }
}

void calEnergies(int *gx, int *gy, int width, int height, int *energies)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = width * y + x;
            energies[idx] = sqrt(gx[idx] * gx[idx] + gy[idx] * gy[idx]);
        }
    }
}

void applySeamCarving(uchar3 *inPixels, int width, int height, int nSeams, uchar3 *&outPixels)
{
    uchar3 *src = inPixels;
    uchar3 *out = (uchar3 *)malloc((width - 1) * height * sizeof(uchar3));
    ;
    // int outHeight = height;
    int srcWidth = width, srcHeight = height;
    float *gx, *gy;
    createSobelFilters(gx, gy);

    float *gaussFilter;
    createGaussianFilter(gaussFilter);

    for (int i = 1; i <= nSeams; i++)
    {
        int outWidth = width - i;
        if (i > 1)
        {
            out = (uchar3 *)realloc(out, outWidth * height * sizeof(uchar3));
        }

        // 1. Convert img to grayscale
        int *grayscaleImg = (int *)malloc(srcWidth * srcHeight * sizeof(int));
        convertRgb2Gray(src, srcWidth, srcHeight, grayscaleImg);

        // 2. Calculate energy value for each pixels: blur --> dx, dy --> energy = |dx| + |dy|
        int *blurImg = (int *)malloc(srcWidth * srcHeight * sizeof(int));
        calConvolution(grayscaleImg, srcWidth, srcHeight, gaussFilter, BLUR_KERNEL_SIZE, blurImg);

        int *dx = (int *)malloc(srcWidth * srcHeight * sizeof(int));
        int *dy = (int *)malloc(srcWidth * srcHeight * sizeof(int));

        calConvolution(blurImg, srcWidth, srcHeight, gx, SOBEL_KERNEL_SIZE, dx);
        calConvolution(blurImg, srcWidth, srcHeight, gy, SOBEL_KERNEL_SIZE, dy);

        int *energy = (int *)malloc(srcWidth * srcHeight * sizeof(int));
        calEnergies(dx, dy, srcWidth, srcHeight, energy);

        // 3. Find seam given energy values above

        int *path = (int *)malloc(srcWidth * srcHeight * sizeof(int));
        int seamIdx = -1;
        findSeam(energy, srcWidth, srcHeight, seamIdx, path);

        // 4. Remove seam
        removeSeam(src, srcWidth, srcHeight, seamIdx, path, out);

        // 5. Reassign variables for next iteration
        src = out;
        srcWidth--;

        free(grayscaleImg);
        free(blurImg);
        free(energy);
        free(path);
        free(dx);
        free(dy);
    }

    outPixels = out;

    // Free allocated memory
    free(gx);
    free(gy);
    free(gaussFilter);
}

int main(int argc, char ** argv) {
    GpuTimer timer;

    int width, height;
    uchar3 *inPixels = NULL;
    uchar3 *outPixels = NULL;

    readPnm(argv[1], width, height, inPixels);
    char * outFileNameBase = strtok(argv[2], ".");
    int nSeams = atoi(argv[3]);

    for (int i = 0; i < 1; i++)
    {

        timer.Start();
        applySeamCarving(inPixels, width, height, nSeams, outPixels);
        timer.Stop();

        printf("Version CPU, %d seams: %f ms\n", nSeams, timer.Elapsed());
        writePnm(outPixels, width - nSeams, height, concatStr(outFileNameBase, "_host.pnm"));
    }
}