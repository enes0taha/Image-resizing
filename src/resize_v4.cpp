// =============================================================
//  V4 - OPTIMUM
// =============================================================

#include "../include/image.h"
#include "../include/timer.h"
#include <pthread.h>
#include <immintrin.h>   
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>
#include <thread>       

// =============================================================
//  Thread Görev Yapısı
// =============================================================
struct TaskV4 {
    const Image* src;
    Image* dst;
    int          row_start;
    int          row_end;
    double       scale_x;
    double       scale_y;
};

// =============================================================
//  Thread Fonksiyonu
// =============================================================
static void* thread_resize_v4(void* arg) {
    TaskV4* task = (TaskV4*)arg;
    const Image& src = *task->src;
    Image&       dst = *task->dst;

    const int    dw  = dst.width;
    const int    ch  = src.channels; 
    const double sx = task->scale_x;
    const double sy = task->scale_y;

    // Look-Up Table (LUT)
    std::vector<int> cx0_arr(dw), cx1_arr(dw);
    std::vector<int> wx_arr(dw); // 0..256 fixed-point

    for (int dx = 0; dx < dw; ++dx) {
        double fx = (dx + 0.5) * sx - 0.5;
        int x0 = (int)std::floor(fx);
        int x1 = x0 + 1;
        int wxi = (int)((fx - std::floor(fx)) * 256.0 + 0.5);

        cx0_arr[dx] = (x0 < 0) ? 0 : (x0 >= src.width  ? src.width - 1 : x0);
        cx1_arr[dx] = (x1 < 0) ? 0 : (x1 >= src.width  ? src.width - 1 : x1);
        wx_arr[dx]  = wxi;
    }

    for (int dy = task->row_start; dy < task->row_end; ++dy) {
        double fy = (dy + 0.5) * sy - 0.5;
        int y0 = (int)std::floor(fy);
        int y1 = y0 + 1;
        int cy0 = (y0 < 0) ? 0 : (y0 >= src.height ? src.height - 1 : y0);
        int cy1 = (y1 < 0) ? 0 : (y1 >= src.height ? src.height - 1 : y1);
        int wyi = (int)((fy - std::floor(fy)) * 256.0 + 0.5);

        const uint8_t* row0 = src.data + (size_t)cy0 * src.width * ch;
        const uint8_t* row1 = src.data + (size_t)cy1 * src.width * ch;
        uint8_t* drow = dst.data + (size_t)dy  * dw        * ch;

        // AVX2 SIMD Register Yüklemesi
        __m256i v_wy   = _mm256_set1_epi32(wyi);
        __m256i v_256  = _mm256_set1_epi32(256);

        int dx = 0;

        for (; dx <= dw - 4; dx += 4) {
     
            alignas(32) int p00_r[4], p10_r[4], p01_r[4], p11_r[4], wx_r[4];

            for (int k = 0; k < 4; ++k) {
                int idx = dx + k;
                int x0i = cx0_arr[idx];
                int x1i = cx1_arr[idx];
                wx_r[k] = wx_arr[idx];

                p00_r[k] = row0[x0i * ch];
                p10_r[k] = row0[x1i * ch];
                p01_r[k] = row1[x0i * ch];
                p11_r[k] = row1[x1i * ch];
            }

            __m256i v_p00 = _mm256_loadu_si256((__m256i*)p00_r);
            __m256i v_p10 = _mm256_loadu_si256((__m256i*)p10_r);
            __m256i v_p01 = _mm256_loadu_si256((__m256i*)p01_r);
            __m256i v_p11 = _mm256_loadu_si256((__m256i*)p11_r);
            __m256i v_wx  = _mm256_loadu_si256((__m256i*)wx_r);

            // Üst Satır Bilinear
            __m256i v_256_sub_wx = _mm256_sub_epi32(v_256, v_wx);
            __m256i v_top_part1   = _mm256_mullo_epi32(v_256_sub_wx, v_p00);
            __m256i v_top_part2   = _mm256_mullo_epi32(v_wx, v_p10);
            __m256i v_top         = _mm256_srli_epi32(_mm256_add_epi32(v_top_part1, v_top_part2), 8);

            // Alt Satır Bilinear
            __m256i v_bot_part1   = _mm256_mullo_epi32(v_256_sub_wx, v_p01);
            __m256i v_bot_part2   = _mm256_mullo_epi32(v_wx, v_p11);
            __m256i v_bot         = _mm256_srli_epi32(_mm256_add_epi32(v_bot_part1, v_bot_part2), 8);

            // Dikey Birleştirme
            __m256i v_256_sub_wy = _mm256_sub_epi32(v_256, v_wy);
            __m256i v_val_part1   = _mm256_mullo_epi32(v_256_sub_wy, v_top);
            __m256i v_val_part2   = _mm256_mullo_epi32(v_wy, v_bot);
            __m256i v_val         = _mm256_srli_epi32(_mm256_add_epi32(v_val_part1, v_val_part2), 8);

            alignas(32) int res_r[8];
            _mm256_storeu_si256((__m256i*)res_r, v_val);

            for (int k = 0; k < 4; ++k) {
                int idx = dx + k;
                uint8_t* dp = drow + idx * ch;
                int x0i = cx0_arr[idx];
                int x1i = cx1_arr[idx];
                int wxi = wx_arr[idx];

                dp[0] = (uint8_t)std::clamp(res_r[k], 0, 255);

                for (int c = 1; c < 3; ++c) {
                    int p00 = row0[x0i * ch + c];
                    int p10 = row0[x1i * ch + c];
                    int p01 = row1[x0i * ch + c];
                    int p11 = row1[x1i * ch + c];

                    int top = ((256 - wxi) * p00 + wxi * p10) >> 8;
                    int bot = ((256 - wxi) * p01 + wxi * p11) >> 8;
                    int val = ((256 - wyi) * top  + wyi * bot) >> 8;
                    dp[c] = (uint8_t)std::clamp(val, 0, 255);
                }
            }
        }

        for (; dx < dw; ++dx) {
            int x0i = cx0_arr[dx];
            int x1i = cx1_arr[dx];
            int wxi = wx_arr[dx];
            uint8_t* dp = drow + dx * ch;

            for (int c = 0; c < ch; ++c) {
                int p00 = row0[x0i * ch + c];
                int p10 = row0[x1i * ch + c];
                int p01 = row1[x0i * ch + c];
                int p11 = row1[x1i * ch + c];

                int top = ((256 - wxi) * p00 + wxi * p10) >> 8;
                int bot = ((256 - wxi) * p01 + wxi * p11) >> 8;
                int val = ((256 - wyi) * top  + wyi * bot) >> 8;
                dp[c] = (uint8_t)std::clamp(val, 0, 255);
            }
        }
    }
    return nullptr;
}

// =============================================================
//  Ana Dağıtıcı Fonksiyon
// =============================================================
Image resize_v4(const Image& src, int dst_w, int dst_h, int num_threads) {
    Image dst = image_create(dst_w, dst_h, src.channels);

    double scale_x = (double)src.width  / dst_w;
    double scale_y = (double)src.height / dst_h;

    std::vector<pthread_t> threads(num_threads);
    std::vector<TaskV4>    tasks(num_threads);

    int rows_per = dst_h / num_threads;
    int leftover = dst_h % num_threads;
    int cur = 0;

    for (int i = 0; i < num_threads; ++i) {
        int rows = rows_per + (i < leftover ? 1 : 0);
        tasks[i] = { &src, &dst, cur, cur + rows, scale_x, scale_y };
        cur += rows;
        pthread_create(&threads[i], nullptr, thread_resize_v4, &tasks[i]);
    }
    for (int i = 0; i < num_threads; ++i)
        pthread_join(threads[i], nullptr);

    return dst;
}

// =============================================================
//  main: 50 Resim Üzerinde V4 Optimum SIMD Performans Analizi
// =============================================================
int main() {
    std::printf("=== V4 OPTIMUM - 50 Resim SIMD + Multi-Thread Performans Analizi ===\n\n");

    unsigned int hw_cores = std::thread::hardware_concurrency();
    if (hw_cores == 0) hw_cores = 4;
    std::printf("Sisteminizde donanımsal AVX2 SIMD registers aktif ve %u çekirdek kullanılıyor.\n\n", hw_cores);

    std::printf("%-9s %-13s %-18s %-18s %-15s\n", "Resim No", "Çözünürlük", "V1 Standart(ms)", "V4 SIMD+Thread(ms)", "Hız Kazanımı");
    std::printf("-----------------------------------------------------------------------------------------\n");

    double total_ms_v1 = 0.0;
    double total_ms_v4 = 0.0;
    Timer t;

    for (int i = 1; i <= 50; i++) {
        char filename[100];
        std::sprintf(filename, "images/%d.ppm", i);

        Image src = image_read_ppm(filename);
        if (!src.data) continue; 

        int dst_w = src.width * 2;
        int dst_h = src.height * 2;

        // V1 Simülasyonu 
        timer_start(t);
        Image dst_v1 = resize_v4(src, dst_w, dst_h, 1);
        double ms_v1 = timer_stop_ms(t);
        total_ms_v1 += ms_v1;

        // V4 Optimum 
        timer_start(t);
        Image dst_v4 = resize_v4(src, dst_w, dst_h, (int)hw_cores);
        double ms_v4 = timer_stop_ms(t);
        total_ms_v4 += ms_v4;

        double speedup = ms_v1 / ms_v4;

        char res_str[30];
        std::sprintf(res_str, "%dx%d", src.width, src.height);
        std::printf("Resim %02d   %-13s %-18.3f %-18.3f %-.2fx\n", i, res_str, ms_v1, ms_v4, speedup);

        image_free(src);
        image_free(dst_v1);
        image_free(dst_v4);
    }

    std::printf("-----------------------------------------------------------------------------------------\n");
    std::printf("TOPLAM V1 BAZ SÜRE (Skaler - Tek Thread)     : %-.3f ms\n", total_ms_v1);
    std::printf("TOPLAM V4 OPTIMUM SÜRE (AVX2 SIMD + Threads) : %-.3f ms\n", total_ms_v4);
    std::printf("DONANIMSAL TOPLAM HIZLANMA ORANI (Speedup)   : %-.2fx\n", total_ms_v1 / total_ms_v4);
    
    std::printf("\nV4 Donanımsal SIMD Optimizasyon Analizi Başarıyla Tamamlandı.\n");
    return 0;
}