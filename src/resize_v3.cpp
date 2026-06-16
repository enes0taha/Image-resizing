// =============================================================
//  V3 - PARALLEL
// =============================================================

#include "../include/image.h"
#include "../include/timer.h"
#include <pthread.h>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>
#include <thread> 

// =============================================================
//  Her thread'e verilecek görev yapısı
// =============================================================
struct ThreadTask {
    const Image* src;
    Image* dst;
    int          row_start; 
    int          row_end;   
    double       scale_x;
    double       scale_y;
};

// =============================================================
//  Bilinear interpolasyon 
// =============================================================
static void* thread_bilinear(void* arg) {
    ThreadTask* task = (ThreadTask*)arg;
    const Image& src = *task->src;
    Image&       dst = *task->dst;

    const double sx = task->scale_x;
    const double sy = task->scale_y;
    const int    ch  = src.channels;
    const int    dw  = dst.width;

    for (int dy = task->row_start; dy < task->row_end; ++dy) {
        const double fy = (dy + 0.5) * sy - 0.5;
        const int    y0 = (int)std::floor(fy);
        const int    y1 = y0 + 1;
        const int    cy0 = (y0 < 0) ? 0 : (y0 >= src.height ? src.height-1 : y0);
        const int    cy1 = (y1 < 0) ? 0 : (y1 >= src.height ? src.height-1 : y1);
        const double wy  = fy - std::floor(fy);

        const uint8_t* row0 = src.data + cy0 * src.width * ch;
        const uint8_t* row1 = src.data + cy1 * src.width * ch;
        uint8_t* drow = dst.data + dy  * dw        * ch;

        for (int dx = 0; dx < dw; ++dx) {
            const double fx = (dx + 0.5) * sx - 0.5;
            const int    x0 = (int)std::floor(fx);
            const int    x1 = x0 + 1;
            const int    cx0 = (x0 < 0) ? 0 : (x0 >= src.width ? src.width-1 : x0);
            const int    cx1 = (x1 < 0) ? 0 : (x1 >= src.width ? src.width-1 : x1);
            const double wx  = fx - std::floor(fx);

            uint8_t* dp = drow + dx * ch;

            for (int c = 0; c < ch; ++c) {
                double p00 = row0[cx0 * ch + c];
                double p10 = row0[cx1 * ch + c];
                double p01 = row1[cx0 * ch + c];
                double p11 = row1[cx1 * ch + c];

                double val = (1.0 - wy) * ((1.0 - wx) * p00 + wx * p10)
                           +        wy  * ((1.0 - wx) * p01 + wx * p11);

                int ival = (int)(val + 0.5);
                dp[c] = (uint8_t)(ival < 0 ? 0 : ival > 255 ? 255 : ival);
            }
        }
    }
    return nullptr;
}

// =============================================================
//  Ana resize fonksiyonu 
// =============================================================
Image resize_parallel(const Image& src, int dst_w, int dst_h, int num_threads) {
    Image dst = image_create(dst_w, dst_h, src.channels);

    double scale_x = (double)src.width  / dst_w;
    double scale_y = (double)src.height / dst_h;

    std::vector<pthread_t>  threads(num_threads);
    std::vector<ThreadTask> tasks(num_threads);

    int rows_per_thread = dst_h / num_threads;
    int leftover        = dst_h % num_threads;

    int current_row = 0;
    for (int i = 0; i < num_threads; ++i) {
        int rows = rows_per_thread + (i < leftover ? 1 : 0);
        tasks[i] = { &src, &dst,
                     current_row, current_row + rows,
                     scale_x, scale_y };
        current_row += rows;

        pthread_create(&threads[i], nullptr, thread_bilinear, &tasks[i]);
    }

    for (int i = 0; i < num_threads; ++i)
        pthread_join(threads[i], nullptr);

    return dst;
}

// =============================================================
//  main: 50 Resim Üzerinde Çoklu Çekirdek Performans Analizi
// =============================================================
int main() {
    std::printf("=== V3 PARALLEL - 50 Resim Pthread Performans Analizi ===\n\n");

    unsigned int hw_cores = std::thread::hardware_concurrency();
    if (hw_cores == 0) hw_cores = 4;
    std::printf("Sisteminizde tespit edilen mantıksal işlemci çekirdeği: %u\n\n", hw_cores);

    std::printf("%-9s %-13s %-15s %-15s %-15s\n", "Resim No", "Çözünürlük", "1 Thread (ms)", "Max Thread(ms)", "Hız Kazanımı");
    std::printf("-------------------------------------------------------------------------\n");

    double total_ms_single = 0.0;
    double total_ms_multi = 0.0;
    Timer t;

    for (int i = 1; i <= 50; i++) {
        char filename[100];
        std::sprintf(filename, "images/%d.ppm", i);

        Image src = image_read_ppm(filename);
        if (!src.data) continue; 

        int dst_w = src.width * 2;
        int dst_h = src.height * 2;

        timer_start(t);
        Image dst_single = resize_parallel(src, dst_w, dst_h, 1);
        double ms_single = timer_stop_ms(t);
        total_ms_single += ms_single;

        timer_start(t);
        Image dst_multi = resize_parallel(src, dst_w, dst_h, (int)hw_cores);
        double ms_multi = timer_stop_ms(t);
        total_ms_multi += ms_multi;

        double speedup = ms_single / ms_multi;

        char res_str[30];
        std::sprintf(res_str, "%dx%d", src.width, src.height);
        std::printf("Resim %02d   %-13s %-15.3f %-15.3f %-.2fx\n", i, res_str, ms_single, ms_multi, speedup);

        image_free(src);
        image_free(dst_single);
        image_free(dst_multi);
    }

    std::printf("-------------------------------------------------------------------------\n");
    std::printf("TOPLAM SERİ SÜRE (Tek İş Parçacığı) : %-.3f ms\n", total_ms_single);
    std::printf("TOPLAM PARALEL SÜRE (%u İş Parçacığı): %-.3f ms\n", hw_cores, total_ms_multi);
    std::printf("GENEL HIZ KAZANIM ORANI (Speedup)   : %-.2fx\n", total_ms_single / total_ms_multi);
    
    std::printf("\nV3 Çoklu Çekirdek Analizi Başarıyla Tamamlandı.\n");
    return 0;
}