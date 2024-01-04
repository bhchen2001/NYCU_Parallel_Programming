#ifndef KERNEL_H
#define KERNEL_H

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage);

#endif