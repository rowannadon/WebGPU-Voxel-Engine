#include <iostream>
#include <vector>
#include <cmath>

#include "stb_image.h"

class ImageUpscaler {
private:
    // Linear interpolation between two values
    float lerp(float a, float b, float t) {
        return a + t * (b - a);
    }

    // Bilinear interpolation for a single channel
    float bilinearInterpolate(const unsigned char* data, int width, int height,
        float x, float y, int channel, int channels) {
        // Clamp coordinates to image bounds
        x = std::max(0.0f, std::min(x, width - 1.0f));
        y = std::max(0.0f, std::min(y, height - 1.0f));

        int x1 = (int)x;
        int y1 = (int)y;
        int x2 = std::min(x1 + 1, width - 1);
        int y2 = std::min(y1 + 1, height - 1);

        float fx = x - x1;
        float fy = y - y1;

        // Get the four surrounding pixels
        float p11 = data[(y1 * width + x1) * channels + channel];
        float p12 = data[(y1 * width + x2) * channels + channel];
        float p21 = data[(y2 * width + x1) * channels + channel];
        float p22 = data[(y2 * width + x2) * channels + channel];

        // Interpolate horizontally
        float top = lerp(p11, p12, fx);
        float bottom = lerp(p21, p22, fx);

        // Interpolate vertically
        return lerp(top, bottom, fy);
    }

public:
    // Structure to hold pixel data
    struct Pixel {
        unsigned char r, g, b, a;

        Pixel() : r(0), g(0), b(0), a(255) {}
        Pixel(unsigned char red, unsigned char green, unsigned char blue, unsigned char alpha = 255)
            : r(red), g(green), b(blue), a(alpha) {
        }
    };

    // Read a pixel from upscaled data at given coordinates
    Pixel readPixel(std::shared_ptr<std::vector<unsigned char>>imageData, int width, int height,
        int channels, int x, int y) {
        // Bounds checking
        if (x < 0 || x >= width || y < 0 || y >= height) {
            std::cerr << "Warning: Pixel coordinates (" << x << ", " << y
                << ") out of bounds for image " << width << "x" << height << std::endl;
            return Pixel(); // Return black pixel with full alpha
        }

        int index = (y * width + x) * channels;
        Pixel pixel;

        switch (channels) {
        case 1: // Grayscale
            pixel.r = pixel.g = pixel.b = imageData->at(index);
            pixel.a = 255;
            break;
        case 3: // RGB
            pixel.r = imageData->at(index);
            pixel.g = imageData->at(index+1);
            pixel.b = imageData->at(index+2);
            pixel.a = 255;
            break;
        case 4: // RGBA
            pixel.r = imageData->at(index);
            pixel.g = imageData->at(index + 1);
            pixel.b = imageData->at(index + 2);
            pixel.a = imageData->at(index + 3);
            break;
        default:
            std::cerr << "Unsupported number of channels: " << channels << std::endl;
            break;
        }

        return pixel;
    }

    // Alternative function that returns raw channel values
    std::vector<unsigned char> readPixelChannels(const std::vector<unsigned char>& imageData,
        int width, int height, int channels, int x, int y) {
        std::vector<unsigned char> channelValues;

        // Bounds checking
        if (x < 0 || x >= width || y < 0 || y >= height) {
            std::cerr << "Warning: Pixel coordinates (" << x << ", " << y
                << ") out of bounds for image " << width << "x" << height << std::endl;
            return channelValues; // Return empty vector
        }

        int index = (y * width + x) * channels;

        for (int c = 0; c < channels; c++) {
            channelValues.push_back(imageData[index + c]);
        }

        return channelValues;
    }

    // Get a single channel value at coordinates
    unsigned char readPixelChannel(const std::vector<unsigned char>& imageData,
        int width, int height, int channels,
        int x, int y, int channel) {
        // Bounds checking
        if (x < 0 || x >= width || y < 0 || y >= height) {
            std::cerr << "Warning: Pixel coordinates (" << x << ", " << y
                << ") out of bounds for image " << width << "x" << height << std::endl;
            return 0;
        }

        if (channel < 0 || channel >= channels) {
            std::cerr << "Warning: Channel " << channel << " out of range for "
                << channels << "-channel image" << std::endl;
            return 0;
        }

        int index = (y * width + x) * channels + channel;
        return imageData[index];
    }

    std::vector<unsigned char> upscaleImage(const unsigned char* originalData,
        int originalWidth, int originalHeight,
        int channels, float scaleFactor) {
        int newWidth = (int)(originalWidth * scaleFactor);
        int newHeight = (int)(originalHeight * scaleFactor);

        std::vector<unsigned char> upscaledData(newWidth * newHeight * channels);

        for (int y = 0; y < newHeight; y++) {
            for (int x = 0; x < newWidth; x++) {
                // Map new coordinates back to original image coordinates
                float originalX = x / scaleFactor;
                float originalY = y / scaleFactor;

                // Interpolate each channel
                for (int c = 0; c < channels; c++) {
                    float interpolatedValue = bilinearInterpolate(
                        originalData, originalWidth, originalHeight,
                        originalX, originalY, c, channels
                    );

                    // Clamp and store the result
                    upscaledData[(y * newWidth + x) * channels + c] =
                        (unsigned char)std::max(0.0f, std::min(255.0f, interpolatedValue));
                }
            }
        }

        return upscaledData;
    }
};