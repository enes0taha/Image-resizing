#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <cstdio>

struct Timer {
    std::chrono::high_resolution_clock::time_point t0, t1;
};

inline void timer_start(Timer& t) {
    t.t0 = std::chrono::high_resolution_clock::now();
}

inline double timer_stop_ms(Timer& t) {
    t.t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t.t1 - t.t0;
    return elapsed.count();
}

inline void timer_print(const char* label, double ms) {
    std::printf("[%-20s]  %.3f ms\n", label, ms);
}

#endif // TIMER_H