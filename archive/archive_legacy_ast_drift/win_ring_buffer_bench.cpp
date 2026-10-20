// Save as: win_ring_buffer_bench.cpp
// Windows Native Shared Memory (CreateFileMapping / MapViewOfFile) Equivalent
// Compile & Run in PowerShell: g++ -O3 -std=c++17 win_ring_buffer_bench.cpp -o win_shm_bench.exe && .\win_shm_bench.exe
#include <iostream>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <windows.h>

// Cache-line aligned (64-byte) static structure per DAG node
struct alignas(64) NodeHistoryRingBuffer {
    float samples[20];       // 80 bytes: Fixed N=20 sliding window
    float running_sum;       // 4 bytes:  O(1) running accumulator
    float nominal_mean;      // 4 bytes:  Design-time offline mean (mu_i)
    uint32_t head;           // 4 bytes:  Current write index
    uint32_t count;          // 4 bytes:  Valid sample count (<= 20)

    void init(float mu_offline) noexcept {
        nominal_mean = mu_offline;
        running_sum = 0.0f;
        head = 0;
        count = 0;
        for (int i = 0; i < 20; ++i) samples[i] = mu_offline;
    }

    inline float update_and_get_adaptive_threshold(float exec_ms, float spatial_base_thresh) noexcept {
        float rolling_mean = (count > 0) ? (running_sum / static_cast<float>(count)) : nominal_mean;
        float drift_norm = (nominal_mean - rolling_mean) / nominal_mean;

        // O(1) Ring Buffer Update (Zero malloc, Zero loops)
        if (count == 20) {
            running_sum -= samples[head];
        } else {
            ++count;
        }
        samples[head] = exec_ms;
        running_sum += exec_ms;
        head = (head + 1 == 20) ? 0 : (head + 1);

        // Clamp feedback adjustment to [-0.20, +0.20]
        float adj = 0.85f * drift_norm;
        if (adj > 0.20f) adj = 0.20f;
        if (adj < -0.20f) adj = -0.20f;

        float eff_thresh = spatial_base_thresh + adj;
        if (eff_thresh < 0.15f) eff_thresh = 0.15f;
        if (eff_thresh > 0.92f) eff_thresh = 0.92f;
        return eff_thresh;
    }
};

int main() {
    const char* map_name = "Local\\rt_dag_history_shm";
    size_t total_size = sizeof(NodeHistoryRingBuffer) * 7;
    HANDLE hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, static_cast<DWORD>(total_size), map_name
    );
    if (!hMapFile) {
        std::cerr << "Failed to create shared memory mapping\n";
        return 1;
    }

    auto* dag_shm = static_cast<NodeHistoryRingBuffer*>(
        MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, total_size)
    );

    dag_shm[0].init(12.5f);
    volatile float sink = 0.0f;

    constexpr int N_ITERS = 1000000;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N_ITERS; ++i) {
        sink = dag_shm[0].update_and_get_adaptive_threshold(11.8f + (i & 3) * 0.2f, 0.55f);
    }
    auto t1 = std::chrono::steady_clock::now();

    double total_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    std::cout << "=== EMBEDDED SHARED MEMORY (Windows Kernel / Named Mapping) BENCHMARK ===\n";
    std::cout << "Memory Footprint per Node : " << sizeof(NodeHistoryRingBuffer) << " bytes (fits in 2 CPU cache lines)\n";
    std::cout << "Total 7-Node DAG Footprint: " << sizeof(NodeHistoryRingBuffer) * 7 << " bytes\n";
    std::cout << "Average O(1) Update Time  : " << (total_ns / N_ITERS) << " nanoseconds per node completion\n";
    std::cout << "Last Effective Threshold  : " << sink << "\n";

    UnmapViewOfFile(dag_shm);
    CloseHandle(hMapFile);
    return 0;
}
