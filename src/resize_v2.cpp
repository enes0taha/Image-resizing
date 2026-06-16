// =============================================================
//  V2 - STANDARD
// =============================================================

#include "../include/image.h"
#include "../include/timer.h"
#include <algorithm>  // std::clamp
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================
//  Yenilen boyutlandırma parametreleri
// =============================================================
enum class InterpMethod { NEAREST, BILINEAR, BICUBIC };

struct ResizeConfig {
    int           dst_w;
    int           dst_h;
    InterpMethod  method;
};

// =============================================================
//  Yardımcı fonksiyonlar
// =============================================================

static inline uint8_t clamp_u8(double v) {
    return (uint8_t)std::clamp((int)(v + 0.5), 0, 255);
}

// Güvenli piksel okuma
static inline uint8_t safe_pixel(const Image& img, int y, int x, int ch) {
    x = std::clamp(x, 0, img.width  - 1);
    y = std::clamp(y, 0, img.height - 1);
    return img.data[(y * img.width + x) * img.channels + ch];
}

// =============================================================
//  Nearest-Neighbor
// =============================================================
static Image resize_nearest(const Image& src, const ResizeConfig& cfg) {
    Image dst = image_create(cfg.dst_w, cfg.dst_h, src.channels);

    const double sx = (double)src.width  / cfg.dst_w;
    const double sy = (double)src.height / cfg.dst_h;

    for (int dy = 0; dy < cfg.dst_h; ++dy) {
        const int src_y = std::clamp((int)(dy * sy), 0, src.height - 1);
        uint8_t* row   = dst.data + dy * cfg.dst_w * src.channels;

        for (int dx = 0; dx < cfg.dst_w; ++dx) {
            const int src_x = std::clamp((int)(dx * sx), 0, src.width - 1);
            const uint8_t* sp = src.data + (src_y * src.width + src_x) * src.channels;
            uint8_t* dp = row + dx * src.channels;
            dp[0] = sp[0];
            dp[1] = sp[1];
            dp[2] = sp[2];
        }
    }
    return dst;
}

// =============================================================
//  Bilinear Interpolation
// =============================================================
static Image resize_bilinear(const Image& src, const ResizeConfig& cfg) {
    Image dst = image_create(cfg.dst_w, cfg.dst_h, src.channels);

    const double sx = (double)src.width  / cfg.dst_w;
    const double sy = (double)src.height / cfg.dst_h;

    for (int dy = 0; dy < cfg.dst_h; ++dy) {
        const double fy = (dy + 0.5) * sy - 0.5;
        const int    y0 = std::clamp((int)floor(fy),     0, src.height - 1);
        const int    y1 = std::clamp((int)floor(fy) + 1, 0, src.height - 1);
        const double wy = fy - floor(fy);

        uint8_t* dst_row = dst.data + dy * cfg.dst_w * src.channels;

        for (int dx = 0; dx < cfg.dst_w; ++dx) {
            const double fx = (dx + 0.5) * sx - 0.5;
            const int    x0 = std::clamp((int)floor(fx),     0, src.width - 1);
            const int    x1 = std::clamp((int)floor(fx) + 1, 0, src.width - 1);
            const double wx = fx - floor(fx);

            uint8_t* dp = dst_row + dx * src.channels;

            for (int ch = 0; ch < src.channels; ++ch) {
                double p00 = src.data[(y0 * src.width + x0) * src.channels + ch];
                double p10 = src.data[(y0 * src.width + x1) * src.channels + ch];
                double p01 = src.data[(y1 * src.width + x0) * src.channels + ch];
                double p11 = src.data[(y1 * src.width + x1) * src.channels + ch];

                dp[ch] = clamp_u8(
                    (1.0 - wy) * ((1.0 - wx) * p00 + wx * p10) +
                           wy  * ((1.0 - wx) * p01 + wx * p11)
                );
            }
        }
    }
    return dst;
}

// =============================================================
//  Bicubic Interpolation (Catmull-Rom kernel)
// =============================================================
static inline double cubic_weight(double t) {
    const double a = -0.5; 
    t = std::fabs(t);
    if (t < 1.0)
        return (a + 2.0) * t*t*t - (a + 3.0) * t*t + 1.0;
    if (t < 2.0)
        return a * t*t*t - 5.0*a * t*t + 8.0*a * t - 4.0*a;
    return 0.0;
}

static Image resize_bicubic(const Image& src, const ResizeConfig& cfg) {
    Image dst = image_create(cfg.dst_w, cfg.dst_h, src.channels);

    const double sx = (double)src.width  / cfg.dst_w;
    const double sy = (double)src.height / cfg.dst_h;

    for (int dy = 0; dy < cfg.dst_h; ++dy) {
        const double fy = (dy + 0.5) * sy - 0.5;
        const int    iy = (int)floor(fy);

        for (int dx = 0; dx < cfg.dst_w; ++dx) {
            const double fx = (dx + 0.5) * sx - 0.5;
            const int    ix = (int)floor(fx);

            double accum[3] = {0.0, 0.0, 0.0};

            for (int m = -1; m <= 2; ++m) {
                double wy = cubic_weight(fy - (iy + m));
                for (int n = -1; n <= 2; ++n) {
                    double wx = cubic_weight(fx - (ix + n));
                    double w  = wx * wy;
                    
                    for (int ch = 0; ch < src.channels; ++ch) {
                        accum[ch] += w * safe_pixel(src, iy + m, ix + n, ch);
                    }
                }
            }

            uint8_t* dp = dst.data + (dy * cfg.dst_w + dx) * src.channels;
            for (int ch = 0; ch < src.channels; ++ch) {
                dp[ch] = clamp_u8(accum[ch]);
            }
        }
    }
    return dst;
}

// =============================================================
//  Dispatch fonksiyonu
// =============================================================
Image resize(const Image& src, const ResizeConfig& cfg) {
    switch (cfg.method) {
        case InterpMethod::NEAREST:  return resize_nearest (src, cfg);
        case InterpMethod::BILINEAR: return resize_bilinear(src, cfg);
        case InterpMethod::BICUBIC:  return resize_bicubic (src, cfg);
        default: return resize_bilinear(src, cfg);
    }
}

// =============================================================
//  main: 50 Resimlik Gerçek Veri Seti Analizi
// =============================================================
int main() {
    std::printf("=== V2 STANDARD - 50 Resim Modüler Performans Analizi ===\n\n");
    std::printf("%-9s %-13s %-13s %-14s %-13s\n", "Resim No", "Çözünürlük", "Nearest(ms)", "Bilinear(ms)", "Bicubic(ms)");
    std::printf("------------------------------------------------------------------------\n");

    Timer t;

    for (int i = 1; i <= 50; i++) {
        char filename[100];
        std::sprintf(filename, "images/%d.ppm", i);

        Image src = image_read_ppm(filename);
        if (!src.data) continue; 

        int dst_w = src.width * 2;
        int dst_h = src.height * 2;
-
        timer_start(t);
        Image dst_nn = resize(src, {dst_w, dst_h, InterpMethod::NEAREST});
        double ms_nn = timer_stop_ms(t);

        timer_start(t);
        Image dst_bl = resize(src, {dst_w, dst_h, InterpMethod::BILINEAR});
        double ms_bl = timer_stop_ms(t);

        double ms_bc = 0.0;
        Image dst_bc = {0, 0, 0, nullptr};
        
        if (i <= 40) {
            timer_start(t);
            dst_bc = resize(src, {dst_w, dst_h, InterpMethod::BICUBIC});
            ms_bc = timer_stop_ms(t);
        }

        char res_str[30];
        std::sprintf(res_str, "%dx%d", src.width, src.height);
        
        if (i <= 40) {
            std::printf("Resim %02d   %-13s %-13.3f %-14.3f %-13.3f\n", i, res_str, ms_nn, ms_bl, ms_bc);
        } else {
            std::printf("Resim %02d   %-13s %-13.3f %-14.3f %-13s\n", i, res_str, ms_nn, ms_bl, "[Ağır Yük-Es Geçildi]");
        }

        image_free(src);
        image_free(dst_nn);
        image_free(dst_bl);
        if (dst_bc.data) image_free(dst_bc);
    }

    std::printf("------------------------------------------------------------------------\n");
    std::printf("\nV2 Modüler C++ Analizi Başarıyla Tamamlandı.\n");
    return 0;
}