#include "../include/image.h"
#include "../include/timer.h"
#include <pthread.h>
#include <immintrin.h>   // V4 AVX2 SIMD
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <thread>        
#include <cstdlib>       

// =============================================================
//  V1 - BASELINE: Skaler Bilinear Interpolation 
// =============================================================
static Image resize_v1(const Image& src, int dw, int dh) {
    Image dst = image_create(dw, dh, src.channels);
    double sx = (double)src.width / dw;
    double sy = (double)src.height / dh;

    for (int dy = 0; dy < dh; ++dy) {
        double fy = (dy + 0.5) * sy - 0.5;
        int y0 = (int)std::floor(fy);
        int y1 = y0 + 1;
        
        if (y0 < 0) y0 = 0; if (y0 >= src.height) y0 = src.height - 1;
        if (y1 < 0) y1 = 0; if (y1 >= src.height) y1 = src.height - 1;
        double wy = fy - std::floor(fy);

        for (int dx = 0; dx < dw; ++dx) {
            double fx = (dx + 0.5) * sx - 0.5;
            int x0 = (int)std::floor(fx);
            int x1 = x0 + 1;
            
            if (x0 < 0) x0 = 0; if (x0 >= src.width) x0 = src.width - 1;
            if (x1 < 0) x1 = 0; if (x1 >= src.width) x1 = src.width - 1;
            double wx = fx - std::floor(fx);

            uint8_t* dp = dst.data + (dy * dw + dx) * src.channels;
            for (int c = 0; c < src.channels; ++c) {
                double p00 = src.data[(y0 * src.width + x0) * src.channels + c];
                double p10 = src.data[(y0 * src.width + x1) * src.channels + c];
                double p01 = src.data[(y1 * src.width + x0) * src.channels + c];
                double p11 = src.data[(y1 * src.width + x1) * src.channels + c];

                double v = (1.0 - wy) * ((1.0 - wx) * p00 + wx * p10) + wy * ((1.0 - wx) * p01 + wx * p11);
                int iv = (int)(v + 0.5);
                dp[c] = (uint8_t)(iv < 0 ? 0 : iv > 255 ? 255 : iv);
            }
        }
    }
    return dst;
}

// =============================================================
//  V2 - STANDARD: Catmull-Rom Bicubic Interpolation
// =============================================================
static inline double cubic_weight(double t) {
    const double a = -0.5; 
    t = std::fabs(t);
    if (t < 1.0) return (a + 2.0) * t*t*t - (a + 3.0) * t*t + 1.0;
    if (t < 2.0) return a * t*t*t - 5.0*a * t*t + 8.0*a * t - 4.0*a;
    return 0.0;
}

static inline uint8_t safe_pixel(const Image& img, int y, int x, int ch) {
    if (x < 0) x = 0; else if (x >= img.width) x = img.width - 1;
    if (y < 0) y = 0; else if (y >= img.height) y = img.height - 1;
    return img.data[(y * img.width + x) * img.channels + ch];
}

static Image resize_v2_bicubic(const Image& src, int dw, int dh) {
    Image dst = image_create(dw, dh, src.channels);
    double sx = (double)src.width / dw;
    double sy = (double)src.height / dh;

    for (int dy = 0; dy < dh; ++dy) {
        double fy = (dy + 0.5) * sy - 0.5;
        int iy = (int)std::floor(fy);

        for (int dx = 0; dx < dw; ++dx) {
            double fx = (dx + 0.5) * sx - 0.5;
            int ix = (int)std::floor(fx);

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
            uint8_t* dp = dst.data + (dy * dw + dx) * src.channels;
            for (int ch = 0; ch < src.channels; ++ch) {
                int iv = (int)(accum[ch] + 0.5);
                dp[ch] = (uint8_t)(iv < 0 ? 0 : iv > 255 ? 255 : iv);
            }
        }
    }
    return dst;
}

// =============================================================
//  V3 - PARALLEL: Pthread Bilinear (Güvenli Sınırlar)
// =============================================================
struct TaskV3 {
    const Image* src; Image* dst;
    int rs, re; double sx, sy;
};

static void* thread_v3_fn(void* arg) {
    TaskV3* t = (TaskV3*)arg;
    const Image& src = *t->src; Image& dst = *t->dst;
    const int dw = dst.width; const int ch = src.channels;

    for (int dy = t->rs; dy < t->re; ++dy) {
        double fy = (dy + 0.5) * t->sy - 0.5;
        int y0 = (int)std::floor(fy);
        int y1 = y0 + 1;
        if (y0 < 0) y0 = 0; if (y0 >= src.height) y0 = src.height - 1;
        if (y1 < 0) y1 = 0; if (y1 >= src.height) y1 = src.height - 1;
        double wy = fy - std::floor(fy);

        const uint8_t* r0 = src.data + y0 * src.width * ch;
        const uint8_t* r1 = src.data + y1 * src.width * ch;
        uint8_t* dr = dst.data + dy * dw * ch;

        for (int dx = 0; dx < dw; ++dx) {
            double fx = (dx + 0.5) * t->sx - 0.5;
            int x0 = (int)std::floor(fx);
            int x1 = x0 + 1;
            if (x0 < 0) x0 = 0; if (x0 >= src.width) x0 = src.width - 1;
            if (x1 < 0) x1 = 0; if (x1 >= src.width) x1 = src.width - 1;
            double wx = fx - std::floor(fx);

            uint8_t* dp = dr + dx * ch;
            for (int c = 0; c < ch; ++c) {
                double p00 = r0[x0 * ch + c]; double p10 = r0[x1 * ch + c];
                double p01 = r1[x0 * ch + c]; double p11 = r1[x1 * ch + c];
                double v = (1.0 - wy) * ((1.0 - wx) * p00 + wx * p10) + wy * ((1.0 - wx) * p01 + wx * p11);
                int iv = (int)(v + 0.5);
                dp[c] = (uint8_t)(iv < 0 ? 0 : iv > 255 ? 255 : iv);
            }
        }
    }
    return nullptr;
}

static Image resize_v3_parallel(const Image& src, int dw, int dh, int nt) {
    Image dst = image_create(dw, dh, src.channels);
    double sx = (double)src.width / dw; double sy = (double)src.height / dh;
    std::vector<pthread_t> th(nt); std::vector<TaskV3> tk(nt);
    int rpt = dh / nt, lft = dh % nt, cur = 0;

    for (int i = 0; i < nt; ++i) {
        int r = rpt + (i < lft ? 1 : 0);
        tk[i] = {&src, &dst, cur, cur + r, sx, sy}; cur += r;
        pthread_create(&th[i], nullptr, thread_v3_fn, &tk[i]);
    }
    for (int i = 0; i < nt; ++i) pthread_join(th[i], nullptr);
    return dst;
}

// =============================================================
//  V4 - OPTIMUM: Pthread + AVX2 Manuel SIMD 
// =============================================================
struct TaskV4 {
    const Image* src; Image* dst;
    int rs, re; double sx, sy;
};

static void* thread_v4_fn(void* arg) {
    TaskV4* task = (TaskV4*)arg;
    const Image& src = *task->src; Image& dst = *task->dst;
    const int dw = dst.width; const int ch = src.channels;

    std::vector<int> cx0_arr(dw), cx1_arr(dw), wx_arr(dw);
    for (int dx = 0; dx < dw; ++dx) {
        double fx = (dx + 0.5) * task->sx - 0.5;
        int x0 = (int)std::floor(fx);
        int x1 = x0 + 1;
        if (x0 < 0) x0 = 0; if (x0 >= src.width) x0 = src.width - 1;
        if (x1 < 0) x1 = 0; if (x1 >= src.width) x1 = src.width - 1;
        cx0_arr[dx] = x0;
        cx1_arr[dx] = x1;
        wx_arr[dx]  = (int)((fx - std::floor(fx)) * 256.0 + 0.5);
    }

    for (int dy = task->rs; dy < task->re; ++dy) {
        double fy = (dy + 0.5) * task->sy - 0.5;
        int y0 = (int)std::floor(fy);
        int y1 = y0 + 1;
        if (y0 < 0) y0 = 0; if (y0 >= src.height) y0 = src.height - 1;
        if (y1 < 0) y1 = 0; if (y1 >= src.height) y1 = src.height - 1;
        int wyi = (int)((fy - std::floor(fy)) * 256.0 + 0.5);

        const uint8_t* row0 = src.data + (size_t)y0 * src.width * ch;
        const uint8_t* row1 = src.data + (size_t)y1 * src.width * ch;
        uint8_t* drow = dst.data + (size_t)dy * dw * ch;

        int dx = 0;
        for (; dx < dw - 4; dx += 4) {
            alignas(32) int p00_r[4], p10_r[4], p01_r[4], p11_r[4], wx_r[4];
            for (int k = 0; k < 4; ++k) {
                int idx = dx + k;
                p00_r[k] = row0[cx0_arr[idx] * ch]; 
                p10_r[k] = row0[cx1_arr[idx] * ch];
                p01_r[k] = row1[cx0_arr[idx] * ch]; 
                p11_r[k] = row1[cx1_arr[idx] * ch];
                wx_r[k]  = wx_arr[idx];
            }

            __m256i v_wy = _mm256_set1_epi32(wyi);
            __m256i v_256 = _mm256_set1_epi32(256);
            __m256i v_p00 = _mm256_loadu_si256((__m256i*)p00_r);
            __m256i v_p10 = _mm256_loadu_si256((__m256i*)p10_r);
            __m256i v_p01 = _mm256_loadu_si256((__m256i*)p01_r);
            __m256i v_p11 = _mm256_loadu_si256((__m256i*)p11_r);
            __m256i v_wx  = _mm256_loadu_si256((__m256i*)wx_r);

            __m256i v_256_sub_wx = _mm256_sub_epi32(v_256, v_wx);
            __m256i v_top = _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi32(v_256_sub_wx, v_p00), _mm256_mullo_epi32(v_wx, v_p10)), 8);
            __m256i v_bot = _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi32(v_256_sub_wx, v_p01), _mm256_mullo_epi32(v_wx, v_p11)), 8);
            __m256i v_val = _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(v_256, v_wy), v_top), _mm256_mullo_epi32(v_wy, v_bot)), 8);

            alignas(32) int res_r[8];
            _mm256_storeu_si256((__m256i*)res_r, v_val);

            for (int k = 0; k < 4; ++k) {
                int idx = dx + k; uint8_t* dp = drow + idx * ch;
                int x0i = cx0_arr[idx]; int x1i = cx1_arr[idx]; int wxi = wx_arr[idx];
                
                int r_val = res_r[k];
                dp[0] = (uint8_t)(r_val < 0 ? 0 : r_val > 255 ? 255 : r_val);

                for (int c = 1; c < ch; ++c) {
                    int p00 = row0[x0i * ch + c]; int p10 = row0[x1i * ch + c];
                    int p01 = row1[x0i * ch + c]; int p11 = row1[x1i * ch + c];
                    int top = ((256 - wxi) * p00 + wxi * p10) >> 8;
                    int bot = ((256 - wxi) * p01 + wxi * p11) >> 8;
                    int final_v = ((256 - wyi) * top + wyi * bot) >> 8;
                    dp[c] = (uint8_t)(final_v < 0 ? 0 : final_v > 255 ? 255 : final_v);
                }
            }
        }
        for (; dx < dw; ++dx) {
            int x0i = cx0_arr[dx]; int x1i = cx1_arr[dx]; int wxi = wx_arr[dx];
            uint8_t* dp = drow + dx * ch;
            for (int c = 0; c < ch; ++c) {
                int p00 = row0[x0i * ch + c]; int p10 = row0[x1i * ch + c];
                int p01 = row1[x0i * ch + c]; int p11 = row1[x1i * ch + c];
                int top = ((256 - wxi) * p00 + wxi * p10) >> 8;
                int bot = ((256 - wxi) * p01 + wxi * p11) >> 8;
                int final_v = ((256 - wyi) * top + wyi * bot) >> 8;
                dp[c] = (uint8_t)(final_v < 0 ? 0 : final_v > 255 ? 255 : final_v);
            }
        }
    }
    return nullptr;
}

static Image resize_v4_optimum(const Image& src, int dw, int dh, int nt) {
    Image dst = image_create(dw, dh, src.channels);
    double sx = (double)src.width / dw; double sy = (double)src.height / dh;
    std::vector<pthread_t> th(nt); std::vector<TaskV4> tk(nt);
    int rpt = dh / nt, lft = dh % nt, cur = 0;

    for (int i = 0; i < nt; ++i) {
        int r = rpt + (i < lft ? 1 : 0);
        tk[i] = {&src, &dst, cur, cur + r, sx, sy}; cur += r;
        pthread_create(&th[i], nullptr, thread_v4_fn, &tk[i]);
    }
    for (int i = 0; i < nt; ++i) pthread_join(th[i], nullptr);
    return dst;
}

// =============================================================
//  Main: Güvenli, Canlı Akışlı ve Dizin Kontrollü Döngü
// =============================================================
int main() {
    std::printf("\n=========================================================================\n");
    std::printf("   IMAGE RESIZING - 50 RESIM DETAYLI KONSOLIDE BENCHMARK RAPORU\n");
    std::printf("=========================================================================\n\n");

    unsigned int hw_cores = std::thread::hardware_concurrency();
    if (hw_cores == 0) hw_cores = 4;
    std::printf("Sistem Aktif Cekirdek Sayisi: %u\n\n", hw_cores);

    std::printf("%-7s %-11s %-9s %-9s %-9s %-9s %-9s\n", "Res No", "Cozunurluk", "V1(ms)", "V2(ms)", "V3-1Th(ms)", "V3-Max(ms)", "V4-SIMD(ms)");
    std::printf("-------------------------------------------------------------------------\n");
    std::fflush(stdout); 

    double total_v1 = 0, total_v2 = 0, total_v3_1 = 0, total_v3_max = 0, total_v4 = 0;
    int processed_images = 0;
    Timer t;

    // Resimlerin yüklenip yüklenmediğini ilk resimden test yeri
    Image test_img = image_read_ppm("images/1.ppm");
    bool use_backup_path = false;
    
    if (!test_img.data) {
        test_img = image_read_ppm("../images/1.ppm");
        if (test_img.data) {
            use_backup_path = true;
            std::printf("[BILGI]: Resimler alt dizinde bulundu, alternatif yol kullaniliyor.\n");
            image_free(test_img);
        } else {
            std::printf("[KRITIK HATA]: 'images/1.ppm' program tarafindan diskte BULUNAMADI!\n");
            std::printf("Lutfen terminalde 'pwd' yazarak su an hangi dizinde oldugunuzu kontrol edin.\n");
            std::fflush(stdout);
            return 1;
        }
    } else {
        image_free(test_img);
    }

    for (int i = 1; i <= 50; i++) {
        char filename[100];
        if (use_backup_path) {
            std::sprintf(filename, "../images/%d.ppm", i);
        } else {
            std::sprintf(filename, "images/%d.ppm", i);
        }

        Image src = image_read_ppm(filename);
        if (!src.data) {
            std::printf("Res %02d   [UYARI: %s okunurken hata olustu!]\n", i, filename);
            std::fflush(stdout);
            continue; 
        }
        processed_images++;

        int dw = src.width * 2;
        int dh = src.height * 2; 

        // V1
        timer_start(t);
        Image d1 = resize_v1(src, dw, dh);
        double m1 = timer_stop_ms(t); total_v1 += m1; image_free(d1);

        // V2 (Bicubic - İlk 20 resim)
        double m2 = 0.0;
        if (i <= 20) {
            timer_start(t);
            Image d2 = resize_v2_bicubic(src, dw, dh);
            m2 = timer_stop_ms(t); total_v2 += m2; image_free(d2);
        }

        // V3 (1 Thread)
        timer_start(t);
        Image d3a = resize_v3_parallel(src, dw, dh, 1);
        double m3_1 = timer_stop_ms(t); total_v3_1 += m3_1; image_free(d3a);

        // V3 (Max Threads)
        timer_start(t);
        Image d3b = resize_v3_parallel(src, dw, dh, (int)hw_cores);
        double m3_m = timer_stop_ms(t); total_v3_max += m3_m; image_free(d3b);

        // V4 (AVX2 SIMD + Max Threads)
        timer_start(t);
        Image d4 = resize_v4_optimum(src, dw, dh, (int)hw_cores);
        double m4 = timer_stop_ms(t); total_v4 += m4; image_free(d4);

        char res_str[20];
        std::sprintf(res_str, "%dx%d", src.width, src.height);

        if (i <= 20) {
            std::printf("Res %02d   %-11s %-9.2f %-9.2f %-9.2f %-9.2f %-9.2f\n", i, res_str, m1, m2, m3_1, m3_m, m4);
        } else {
            std::printf("Res %02d   %-11s %-9.2f %-9s %-9.2f %-9.2f %-9.2f\n", i, res_str, m1, "[Pas-Yuk]", m3_1, m3_m, m4);
        }
        
        std::fflush(stdout); 
        image_free(src);
    }

    std::printf("-------------------------------------------------------------------------\n");
    std::printf("ORTALAMA SÜRELER VE GENEL HIZLANMA ORANLARI (SPEEDUP TABLOSU):\n");
    std::printf("-------------------------------------------------------------------------\n");
    std::printf("| %-25s | %-15s | %-15s |\n", "Versiyon Grubu", "Ortalama Süre", "Hizlanma Çarpani");
    std::printf("-------------------------------------------------------------------------\n");
    std::printf("| %-25s | %-12.2f ms | %-15s |\n", "V1 Baseline (Bilinear)", total_v1 / processed_images, "1.00x (Referans)");
    std::printf("| %-25s | %-12.2f ms | %-15.2fx |\n", "V2 Standard (Bicubic)", total_v2 / (processed_images > 20 ? 20 : processed_images), (total_v1 / processed_images) / (total_v2 / (processed_images > 20 ? 20 : processed_images)));
    std::printf("| %-25s | %-12.2f ms | %-15.2fx |\n", "V3 Single Thread", total_v3_1 / processed_images, total_v1 / total_v3_1);
    std::printf("| %-25s | %-12.2f ms | %-15.2fx |\n", "V3 Multi-Thread Max", total_v3_max / processed_images, total_v1 / total_v3_max);
    std::printf("| %-25s | %-12.2f ms | %-15.2fx |\n", "V4 SIMD + Thread (Optimum)", total_v4 / processed_images, total_v1 / total_v4);
    std::printf("-------------------------------------------------------------------------\n");
    std::fflush(stdout);

    return 0;
}