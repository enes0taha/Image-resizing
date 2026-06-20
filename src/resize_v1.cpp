// =============================================================
//  V1 - BASELINE
// =============================================================

#include "../include/image.h"
#include "../include/timer.h"
#include <cstdio>
#include <cmath>

// -------------------------------------------------------------
//  Nearest-Neighbor yeniden boyutlandırma
// -------------------------------------------------------------
Image resize_nearest(const Image& src, int new_w, int new_h) {
    Image dst = image_create(new_w, new_h, src.channels);

    double scale_x = (double)src.width  / new_w;
    double scale_y = (double)src.height / new_h;

    for (int dy = 0; dy < new_h; dy++) {
        for (int dx = 0; dx < new_w; dx++) {
            int sx = (int)(dx * scale_x);
            int sy = (int)(dy * scale_y);
            
            // Sınır kontrolü
            if (sx >= src.width)  sx = src.width  - 1;
            if (sy >= src.height) sy = src.height - 1;

            for (int ch = 0; ch < src.channels; ch++) {
                pixel(dst, dy, dx, ch) = pixel_c(src, sy, sx, ch);
            }
        }
    }
    return dst;
}

// -------------------------------------------------------------
//  Bilinear Interpolation yeniden boyutlandırma
//  4 komşu pikselin ağırlıklı ortalaması
// -------------------------------------------------------------
Image resize_bilinear(const Image& src, int new_w, int new_h) {
    Image dst = image_create(new_w, new_h, src.channels);

    double scale_x = (double)src.width  / new_w;
    double scale_y = (double)src.height / new_h;

    for (int dy = 0; dy < new_h; dy++) {
        for (int dx = 0; dx < new_w; dx++) {
            double fx = (dx + 0.5) * scale_x - 0.5;
            double fy = (dy + 0.5) * scale_y - 0.5;

            int x0 = (int)std::floor(fx); 
            int y0 = (int)std::floor(fy); 
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            if (x0 < 0) x0 = 0;
            if (y0 < 0) y0 = 0;
            if (x1 >= src.width)  x1 = src.width  - 1;
            if (y1 >= src.height) y1 = src.height - 1;

            // Enterpolasyon ağırlıkları
            double wx = fx - std::floor(fx); 
            double wy = fy - std::floor(fy); 

            for (int ch = 0; ch < src.channels; ch++) {
                double p00 = pixel_c(src, y0, x0, ch);
                double p10 = pixel_c(src, y0, x1, ch);
                double p01 = pixel_c(src, y1, x0, ch);
                double p11 = pixel_c(src, y1, x1, ch);

                double val = (1.0 - wy) * ((1.0 - wx) * p00 + wx * p10)
                           +        wy   * ((1.0 - wx) * p01 + wx * p11);

                int ival = (int)(val + 0.5);
                if (ival < 0)   ival = 0;
                if (ival > 255) ival = 255;
                
                pixel(dst, dy, dx, ch) = (uint8_t)ival;
            }
        }
    }
    return dst;
}

// -------------------------------------------------------------
//  main: 50 Adet Gerçek Resim Üzerinde Test Senaryosu
// -------------------------------------------------------------
int main() {
    std::printf("=== V1 BASELINE - 50 Resim Performans Analizi ===\n\n");
    std::printf("%-10s %-15s %-15s %-15s\n", "Resim No", "Çözünürlük", "Nearest (ms)", "Bilinear (ms)");
    std::printf("-------------------------------------------------------------\n");

    double total_ms_nn = 0.0;
    double total_ms_bl = 0.0;
    Timer t;

    for (int i = 1; i <= 50; i++) {
        char filename[100];
        std::sprintf(filename, "images/%d.ppm", i);

        Image src = image_read_ppm(filename);
        if (!src.data) {
            continue; 
        }

        int out_w = src.width * 2;
        int out_h = src.height * 2;

        // --- Nearest-Neighbor Ölçümü ---
        timer_start(t);
        Image dst_nn = resize_nearest(src, out_w, out_h);
        double ms_nn = timer_stop_ms(t);
        total_ms_nn += ms_nn;

        // --- Bilinear Ölçümü ---
        timer_start(t);
        Image dst_bl = resize_bilinear(src, out_w, out_h);
        double ms_bl = timer_stop_ms(t);
        total_ms_bl += ms_bl;

        char res_str[30];
        std::sprintf(res_str, "%dx%d", src.width, src.height);
        std::printf("Resim %02d   %-15s %-15.3f %-15.3f\n", i, res_str, ms_nn, ms_bl);

        if (i == 1 || i == 50) {
            char out_nn_name[100], out_bl_name[100];
            std::sprintf(out_nn_name, "out_v1_nearest_%d.ppm", i);
            std::sprintf(out_bl_name, "out_v1_bilinear_%d.ppm", i);
            
            image_write_ppm(dst_nn, out_nn_name);
            image_write_ppm(dst_bl, out_bl_name);
        }

        image_free(src);
        image_free(dst_nn);
        image_free(dst_bl);
    }

    std::printf("-------------------------------------------------------------\n");
    std::printf("TOPLAM SÜRE:            %-15.3f %-15.3f ms\n", total_ms_nn, total_ms_bl);
    std::printf("Ortalama Performans Oranı: %.2fx (Bilinear daha yavaş)\n", total_ms_bl / total_ms_nn);
    std::printf("\nV1 Analizi Başarıyla Tamamlandı.\n");
    
    return 0;
}