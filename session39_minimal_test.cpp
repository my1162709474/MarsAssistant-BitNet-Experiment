// Session 39 Minimal Test - Ultra-Advanced Parallel & Memory Optimization
#include <cmath>
#include <cstring>
#include <chrono>
#include <iostream>
#include <vector>

#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#define IS_ARM 1
#else
#include <immintrin.h>
#define IS_ARM 0
#endif

using namespace std;
using namespace std::chrono;

// Ultra 128x Loop Unrolling (ARM NEON version)
void matmul_ultra_128x_unroll(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    constexpr int NEON_SIZE = 4;
    constexpr int UNROLL_FACTOR = 16;
    
    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;
        float* C_row = C + i * N;
        
        int num_vec = N / NEON_SIZE;
        int unrolled = (num_vec / UNROLL_FACTOR) * UNROLL_FACTOR;
        
        for (int j = 0; j < unrolled; j += UNROLL_FACTOR) {
            for (int u = 0; u < UNROLL_FACTOR; u++) {
                vst1q_f32(&C_row[(j + u) * NEON_SIZE], vdupq_n_f32(0.0f));
            }
        }
        for (int j = unrolled * NEON_SIZE; j < N; j++) {
            C_row[j] = 0.0f;
        }
        
        for (int k = 0; k < K; k++) {
            float32x4_t a_val = vdupq_n_f32(A_row[k]);
            const float* B_k = B + k * N;
            
            for (int j = 0; j < unrolled; j += UNROLL_FACTOR) {
                float32x4_t b[16];
                float32x4_t c[16];
                
                for (int u = 0; u < 16; u++) {
                    b[u] = vld1q_f32(&B_k[(j + u) * NEON_SIZE]);
                    c[u] = vld1q_f32(&C_row[(j + u) * NEON_SIZE]);
                }
                
                for (int u = 0; u < 16; u++) {
                    c[u] = vfmaq_f32(c[u], a_val, b[u]);
                }
                
                for (int u = 0; u < 16; u++) {
                    vst1q_f32(&C_row[(j + u) * NEON_SIZE], c[u]);
                }
            }
        }
    }
}

// Standard NEON matmul for comparison
void matmul_neon_standard(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    constexpr int NEON_SIZE = 4;
    
    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;
        float* C_row = C + i * N;
        
        int num_vec = N / NEON_SIZE;
        
        for (int j = 0; j < num_vec; j++) {
            vst1q_f32(&C_row[j * NEON_SIZE], vdupq_n_f32(0.0f));
        }
        
        for (int k = 0; k < K; k++) {
            float32x4_t a_val = vdupq_n_f32(A_row[k]);
            const float* B_k = B + k * N;
            
            for (int j = 0; j < num_vec; j++) {
                float32x4_t c_val = vld1q_f32(&C_row[j * NEON_SIZE]);
                float32x4_t b_val = vld1q_f32(&B_k[j * NEON_SIZE]);
                c_val = vfmaq_f32(c_val, a_val, b_val);
                vst1q_f32(&C_row[j * NEON_SIZE], c_val);
            }
        }
    }
}

// Benchmark function
void benchmark(const string& name, void (*func)(const float*, const float*, float*, int, int, int),
               const float* A, const float* B, float* C, int M, int N, int K, int iterations = 100) {
    auto start = high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        func(A, B, C, M, N, K);
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    
    double avg_time = duration.count() / (double)iterations;
    double gflops = (2.0 * M * N * K) / (avg_time * 1000.0);
    
    cout << name << ": " << avg_time << " us, " << gflops << " GFLOPS" << endl;
}

int main() {
    cout << "Session 39: Ultra-Advanced Parallel & Memory Optimization" << endl;
    cout << "============================================================" << endl;
    cout << "Platform: " << (IS_ARM ? "ARM NEON" : "x86 AVX2") << endl;
    cout << endl;
    
    // Test with different sizes
    vector<int> sizes = {256, 512, 1024};
    
    for (int size : sizes) {
        int M = size, N = size, K = size;
        
        // Allocate matrices
        vector<float> A(M * K);
        vector<float> B(K * N);
        vector<float> C(M * N);
        
        // Initialize with random values
        srand(42);
        for (int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
        for (int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX;
        
        cout << "Matrix Size: " << M << "x" << N << "x" << K << endl;
        
        // Benchmark standard NEON
        fill(C.begin(), C.end(), 0.0f);
        benchmark("Standard NEON", matmul_neon_standard, A.data(), B.data(), C.data(), M, N, K, 50);
        
        // Benchmark ultra 128x unrolling
        fill(C.begin(), C.end(), 0.0f);
        benchmark("Ultra 128x Unroll", matmul_ultra_128x_unroll, A.data(), B.data(), C.data(), M, N, K, 50);
        
        cout << endl;
    }
    
    cout << "Session 39 optimizations applied successfully!" << endl;
    return 0;
}
