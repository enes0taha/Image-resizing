#ifndef IMAGE_H
#define IMAGE_H

#include <cstdint>
#include <cstdlib>
#include <cstdio>

struct Image {
    int width;
    int height;
    int channels;
    uint8_t* data;
};

// Piksellere kolay erişim makroları/fonksiyonları
inline uint8_t& pixel(Image& img, int r, int c, int ch) {
    return img.data[(r * img.width + c) * img.channels + ch];
}

inline uint8_t pixel_c(const Image& img, int r, int c, int ch) {
    return img.data[(r * img.width + c) * img.channels + ch];
}

inline Image image_create(int w, int h, int channels = 3) {
    Image img;
    img.width = w;
    img.height = h;
    img.channels = channels;
    img.data = (uint8_t*)std::malloc(w * h * channels);
    return img;
}

inline void image_free(Image& img) {
    if (img.data) {
        std::free(img.data);
        img.data = nullptr;
    }
}

inline Image image_read_ppm(const char* filename) {
    std::FILE* f = std::fopen(filename, "rb");
    if (!f) {
        std::printf("[HATA] Dosya acilamadi: %s\n", filename);
        return {0, 0, 0, nullptr};
    }
    
    char p, six;
    int w, h, max_val;
    if (std::fscanf(f, "%c%c\n", &p, &six) != 2 || p != 'P' || six != '6') {
        std::printf("[HATA] Gecersiz PPM formati (P6 olmali): %s\n", filename);
        std::fclose(f);
        return {0, 0, 0, nullptr};
    }

    int ch = std::fgetc(f);
    while (ch == '#') {
        while ((ch = std::fgetc(f)) != '\n' && ch != EOF);
        ch = std::fgetc(f);
    }
    std::ungetc(ch, f);

    if (std::fscanf(f, "%d %d\n%d\n", &w, &h, &max_val) != 3) {
        std::fclose(f);
        return {0, 0, 0, nullptr};
    }

    Image img = image_create(w, h, 3);
    std::fread(img.data, 1, w * h * 3, f);
    std::fclose(f);
    return img;
}

inline bool image_write_ppm(const Image& img, const char* filename) {
    std::FILE* f = std::fopen(filename, "wb");
    if (!f) return false;

    std::fprintf(f, "P6\n%d %d\n255\n", img.width, img.height);
    std::fwrite(img.data, 1, img.width * img.height * img.channels, f);
    std::fclose(f);
    return true;
}

#endif // IMAGE_H