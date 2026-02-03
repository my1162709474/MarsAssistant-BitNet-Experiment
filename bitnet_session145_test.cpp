// Session 145 Performance Optimization Test
// Focus: Ultra-Extreme Optimizations

#include <cmath>
#include <cstring>
#include <cfloat>
#include <immintrin.h>
#include <pthread.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>

// Compiler hints
#ifdef __GNUC__
#define HOT_FUNC __attribute__((hot))
#define ALIGNED __attribute__((aligned(32)))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define RESTRICT __restrict__
#define FORCE_INLINE inline __attribute__((always_inline))
#define PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 3)
#else
#define FORCE_INLINE inline
#define PREFETCH_READ(addr)
#define RESTRICT
#endif

// Global counters
static std::atomic<size_t> session145_ops{0};
static std::atomic<size_t> session145_matmul_ops{0};

// ==================== Ultra 512x Loop Unrolling ====================

FORCE_INLINE void matmul_512x_ultra_unroll_avx2(const float* RESTRICT A,
                                                 const float* RESTRICT B,
                                                 float* RESTRICT C,
                                                 int M, int N, int K) {
    constexpr int AVX_SIZE = 8;
    constexpr int UNROLL_N = 64;
    constexpr int UNROLL_K = 16;

    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;

        for (int j = 0; j + UNROLL_N <= N; j += UNROLL_N) {
            __m256 acc[UNROLL_N / AVX_SIZE];
            for (int aj = 0; aj < UNROLL_N / AVX_SIZE; aj++) {
                acc[aj] = _mm256_setzero_ps();
            }

            for (int kk = 0; kk < K; kk += UNROLL_K) {
                int k_end = std::min(kk + UNROLL_K, K);

                for (int k = kk; k < k_end; k++) {
                    PREFETCH_READ(A_row + k + 16);
                    __m256 a_val = _mm256_set1_ps(A_row[k]);

                    for (int bj = 0; bj < UNROLL_N; bj += AVX_SIZE) {
                        __m256 b_vec = _mm256_loadu_ps(B + k * N + j + bj);
                        acc[bj / AVX_SIZE] = _mm256_fmadd_ps(a_val, b_vec, acc[bj / AVX_SIZE]);
                    }
                }
            }

            for (int bj = 0; bj < UNROLL_N; bj += AVX_SIZE) {
                __m256 c_vec = _mm256_loadu_ps(C + i * N + j + bj);
                _mm256_storeu_ps(C + i * N + j + bj, _mm256_add_ps(c_vec, acc[bj / AVX_SIZE]));
            }
        }

        for (int j = (N / UNROLL_N) * UNROLL_N; j < N; j += AVX_SIZE) {
            int remaining = std::min(AVX_SIZE, N - j);
            __m256 acc = _mm256_setzero_ps();

            for (int k = 0; k < K; k++) {
                __m256 a_val = _mm256_set1_ps(A_row[k]);
                __m256 b_vec = _mm256_loadu_ps(B + k * N + j);
                acc = _mm256_fmadd_ps(a_val, b_vec, acc);
            }

            __m256 c_vec = _mm256_loadu_ps(C + i * N + j);
            _mm256_storeu_ps(C + i * N + j, _mm256_add_ps(c_vec, acc));
        }
    }

    session145_matmul_ops.fetch_add(M * N * K);
}

FORCE_INLINE void matmul_avx2(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int AVX_SIZE = 8;

    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;

        for (int j = 0; j < N; j += AVX_SIZE) {
            __m256 acc = _mm256_setzero_ps();

            for (int k = 0; k < K; k++) {
                __m256 a_val = _mm256_set1_ps(A_row[k]);
                __m256 b_vec = _mm256_loadu_ps(B + k * N + j);
                acc = _mm256_fmadd_ps(a_val, b_vec, acc);
            }

            __m256 c_vec = _mm256_loadu_ps(C + i * N + j);
            _mm256_storeu_ps(C + i * N + j, _mm256_add_ps(c_vec, acc));
        }
    }
}

// ==================== Aggressive Prefetch + Cache Blocking ====================

FORCE_INLINE void matmul_cache_blocking_aggressive(const float* RESTRICT A,
                                                    const float* RESTRICT B,
                                                    float* RESTRICT C,
                                                    int M, int N, int K) {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 32;
    constexpr int AVX_SIZE = 8;
    constexpr int PREFETCH_DIST = 64;

    for (int mb = 0; mb < M; mb += BLOCK_M) {
        int mb_end = std::min(mb + BLOCK_M, M);

        for (int nb = 0; nb < N; nb += BLOCK_N) {
            int nb_end = std::min(nb + BLOCK_N, N);

            for (int kb = 0; kb < K; kb += BLOCK_K) {
                int kb_end = std::min(kb + BLOCK_K, K);

                for (int k = kb; k < kb_end; k++) {
                    for (int n = nb; n < nb_end && n < nb + 16; n += AVX_SIZE) {
                        PREFETCH_READ(B + k * N + n + PREFETCH_DIST);
                    }
                }

                for (int i = mb; i < mb_end; i++) {
                    const float* A_row = A + i * K;

                    for (int j = nb; j < nb_end; j += AVX_SIZE) {
                        __m256 acc = _mm256_setzero_ps();

                        for (int k = kb; k < kb_end; k++) {
                            __m256 a_val = _mm256_set1_ps(A_row[k]);
                            __m256 b_vec = _mm256_loadu_ps(B + k * N + j);
                            acc = _mm256_fmadd_ps(a_val, b_vec, acc);
                        }

                        __m256 c_vec = _mm256_loadu_ps(C + i * N + j);
                        _mm256_storeu_ps(C + i * N + j, _mm256_add_ps(c_vec, acc));
                    }
                }
            }
        }
    }
}

// ==================== Hyper-Parallel Matrix Multiply ====================

FORCE_INLINE void matmul_hyper_parallel(const float* A,
                                         const float* B,
                                         float* C,
                                         int M, int N, int K) {
    constexpr int CHUNK_SIZE = 32;
    constexpr int NUM_THREADS = std::thread::hardware_concurrency();

    std::vector<std::thread> threads;
    std::atomic<int> next_row{0};

    auto worker = [&](int thread_id) {
        int local_ops = 0;

        while (true) {
            int start_row = next_row.fetch_add(CHUNK_SIZE, std::memory_order_relaxed);
            if (start_row >= M) break;

            int end_row = std::min(start_row + CHUNK_SIZE, M);

            for (int i = start_row; i < end_row; i++) {
                const float* A_row = A + i * K;
                for (int j = 0; j < N; j += 8) {
                    __m256 acc = _mm256_setzero_ps();
                    for (int k = 0; k < K; k++) {
                        __m256 a_val = _mm256_set1_ps(A_row[k]);
                        __m256 b_vec = _mm256_loadu_ps(B + k * N + j);
                        acc = _mm256_fmadd_ps(a_val, b_vec, acc);
                    }
                    __m256 c_vec = _mm256_loadu_ps(C + i * N + j);
                    _mm256_storeu_ps(C + i * N + j, _mm256_add_ps(c_vec, acc));
                }
                local_ops += N * K;
            }
        }

        session145_ops.fetch_add(local_ops);
    };

    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back(worker, t);
    }

    for (auto& t : threads) {
        t.join();
    }
}

int main() {
    std::cout << "BitNet Session 145 Performance Test" << std::endl;
    std::cout << "====================================" << std::endl;

    int M = 512, N = 512, K = 512;
    int iterations = 100;

    std::cout << "Matrix size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;

    // Allocate aligned memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    posix_memalign((void**)&A, 64, M * K * sizeof(float));
    posix_memalign((void**)&B, 64, K * N * sizeof(float));
    posix_memalign((void**)&C, 64, M * N * sizeof(float));

    // Initialize
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * K; i++) A[i] = dist(rng);
    for (int i = 0; i < K * N; i++) B[i] = dist(rng);
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    // Test methods
    std::vector<std::pair<std::string, std::function<void()>>> methods = {
        {"matmul_avx2", [&]() {
            std::fill(C, C + M * N, 0.0f);
            matmul_avx2(A, B, C, M, N, K);
        }},
        {"matmul_512x_ultra_unroll_avx2", [&]() {
            std::fill(C, C + M * N, 0.0f);
            matmul_512x_ultra_unroll_avx2(A, B, C, M, N, K);
        }},
        {"matmul_cache_blocking_aggressive", [&]() {
            std::fill(C, C + M * N, 0.0f);
            matmul_cache_blocking_aggressive(A, B, C, M, N, K);
        }},
        {"matmul_hyper_parallel", [&]() {
            std::fill(C, C + M * N, 0.0f);
            matmul_hyper_parallel(A, B, C, M, N, K);
        }},
    };

    for (const auto& [name, func] : methods) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; iter++) {
            func();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        double avg_time = static_cast<double>(duration) / iterations;
        double gflops = 2.0 * M * N * K / (avg_time * 1000.0);

        std::cout << name << ": " << avg_time << " us (" << gflops << " GFLOPS)" << std::endl;
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
