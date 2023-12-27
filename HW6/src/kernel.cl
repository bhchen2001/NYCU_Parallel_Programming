__kernel void convolution(int filterWidth, __global float *filter, int imageHeight, int imageWidth, __global float *inputImage, __global float *outputImage){
    int x = get_global_id(0);
    int y = get_global_id(1);
    int halfWidth = filterWidth / 2;
    float sum = 0.0f;
    // for(int i = -halfWidth; i <= halfWidth; i++){
    //     for(int j = -halfWidth; j <= halfWidth; j++){
    //         if(x + i >= 0 && x + i < imageWidth && y + j >= 0 && y + j < imageHeight){
    //             sum += filter[(i + halfWidth) + filterWidth * (j + halfWidth)] * inputImage[(y + j) * imageWidth + (x + i)];
    //         }
    //     }
    // }

    for(int i = 0; i < filterWidth; i++){
        int xIndex = x + i - halfWidth;
        for(int j = 0; j < filterWidth; j++){
            int yIndex = y + j - halfWidth;
            if(xIndex >= 0 && xIndex < imageWidth && yIndex >= 0 && yIndex < imageHeight){
                sum += filter[i + filterWidth * j] * inputImage[yIndex * imageWidth + xIndex];
            }
        }
    }
    outputImage[(y) * imageWidth + (x)] = sum;
}
