__kernel void convolution(int filterWidth, int halfWidth,  __constant const float *filter, __global const float *inputImage, __global float *outputImage){
    int x = get_global_id(0);
    int y = get_global_id(1);

    int imageWidth = get_global_size(0);
    int imageHeight = get_global_size(1);
    // int halfWidth = filterWidth / 2;
    float sum = 0.0f;

    int x_img_start = x - halfWidth < 0 ? 0 : x - halfWidth;
    int x_img_end = x + filterWidth - halfWidth - 1 >= imageWidth ? imageWidth - 1 : x + filterWidth - halfWidth - 1;
    int y_img_start = y - halfWidth < 0 ? 0 : y - halfWidth;
    int y_img_end = y + filterWidth - halfWidth - 1 >= imageHeight ? imageHeight - 1 : y + filterWidth - halfWidth - 1;
    int x_filter_start = x - halfWidth < 0 ? halfWidth - x : 0;
    int x_filter_end = x + filterWidth - halfWidth - 1 >= imageWidth ? filterWidth - 1 - (x + filterWidth - halfWidth - 1 - imageWidth) : filterWidth - 1;
    int y_filter_start = y - halfWidth < 0 ? halfWidth - y : 0;
    int y_filter_end = y + filterWidth - halfWidth - 1 >= imageHeight ? filterWidth - 1 - (y + filterWidth - halfWidth - 1 - imageHeight) : filterWidth - 1;

    for(int j = y_img_start; j <= y_img_end; j++){
        int filter_index = (j - y_img_start + y_filter_start) * filterWidth;
        int image_index = j * imageWidth;
        for(int i = x_img_start; i <= x_img_end; i++){
            if(!filter[i - x_img_start + x_filter_start + filter_index]){
                continue;
            }
            sum += filter[i - x_img_start + x_filter_start + filter_index] * inputImage[image_index + i];
        }
    }

    // for(int i = 0; i < filterWidth; i++){
    //     int xIndex = x + i - halfWidth;
    //     for(int j = 0; j < filterWidth; j++){
    //         int yIndex = y + j - halfWidth;
    //         if(xIndex >= 0 && xIndex < imageWidth && yIndex >= 0 && yIndex < imageHeight){
    //             sum += filter[i + filterWidth * j] * inputImage[yIndex * imageWidth + xIndex];
    //         }
    //     }
    // }

    outputImage[(y) * imageWidth + (x)] = sum;
}
