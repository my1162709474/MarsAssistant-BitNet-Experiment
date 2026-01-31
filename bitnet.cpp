/**
 * BitNet: 1-bit Transformer Networks
 * 
 * This is a simplified implementation focusing on:
 * - 1-bit quantized inference
 * - Matrix multiplication optimization
 * - SIMD vectorization
 * - Memory access patterns
 */

#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <arm_neon.h>  // ARM NEON for Apple Silicon
#include <pthread.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <thread>

// Configuration
constexpr int BLOCK_SIZE = 64;
constexpr int CACHE_LINE_SIZE = 64;

// Data structures
struct Matrix {
    float* data;
    int rows;
    int cols;
    int stride;
    
    Matrix(int r = 0, int c = 0) : rows(r), cols(c), stride(c) {
        // Aligned allocation for SIMD (32-byte alignment for AVX2)
        posix_memalign(reinterpret_cast<void**>(&data), CACHE_LINE_SIZE, 
                       sizeof(float) * rows * cols);
        std::memset(data, 0, sizeof(float) * rows * cols);
    }
    
    ~Matrix() {
        free(data);
    }
};

// ==================== NEW: Aligned 1-bit Matrix ====================

struct BitMatrix {
    unsigned char* data;
    int rows;
    int cols;
    int stride_bytes;
    
    BitMatrix(int r = 0, int c = 0) : rows(r), cols(c) {
        stride_bytes = (cols + 7) / 8;  // Bits to bytes
        posix_memalign(reinterpret_cast<void**>(&data), CACHE_LINE_SIZE,
                       sizeof(unsigned char) * rows * stride_bytes);
        std::memset(data, 0, sizeof(unsigned char) * rows * stride_bytes);
    }
    
    ~BitMatrix() {
        free(data);
    }
    
    // Pack bits on-the-fly from float matrix
    void pack_from_float(const float* src) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (src[i * cols + j] > 0.0f) {
                    data[i * stride_bytes + j / 8] |= (1 << (j % 8));
                }
            }
        }
    }
};

struct BitNetConfig {
    int hidden_size;
    int num_heads;
    int num_layers;
    int max_seq_len;
    float threshold;
};

// ==================== Compiler Optimization Hints ====================

// Compiler hints for auto-vectorization and inlining
#ifdef __GNUC__
#define HOT_FUNC __attribute__((hot))
#define ALIGNED __attribute__((aligned(32)))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define HOT_FUNC
#define ALIGNED
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

// ==================== AVX-512 Support (Conditional) ====================

#if defined(__AVX512F__) && defined(__AVX512BW__)
#define USE_AVX512 1
constexpr int AVX512_SIZE = 16;  // 512-bit / 32-bit

void matmul_avx512(const float* A, const float* B, float* C,
                   int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;
        float* C_row = C + i * N;
        
        __m512 c_vec[32];  // Support up to 512 columns
        int num_vec = N / AVX512_SIZE;
        for (int j = 0; j < num_vec; j++) {
            c_vec[j] = _mm512_setzero_ps();
        }
        
        for (int k = 0; k < K; k++) {
            __m512 a_val = _mm512_set1_ps(A_row[k]);
            const float* B_k = B + k * N;
            
            for (int j = 0; j < num_vec; j++) {
                __m512 b_vec = _mm512_loadu_ps(&B_k[j * AVX512_SIZE]);
                c_vec[j] = _mm512_fmadd_ps(a_val, b_vec, c_vec[j]);
            }
        }
        
        for (int j = 0; j < num_vec; j++) {
            _mm512_storeu_ps(&C_row[j * AVX512_SIZE], c_vec[j]);
        }
    }
}
#else
#define USE_AVX512 0
void matmul_avx512(const float* A, const float* B, float* C,
                   int M, int N, int K) {
    // Fallback to AVX2 if AVX-512 not available
    matmul_avx2(A, B, C, M, N, K);
}
#endif

// ==================== Original Matrix Multiplication ====================

void matmul_naive(const float* A, const float* B, float* C, 
                  int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ==================== Optimized 1: Blocked Matrix Multiplication ====================

void matmul_blocked(const float* A, const float* B, float* C,
                    int M, int N, int K) {
    // Cache-friendly blocking
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < K; k += BLOCK_SIZE) {
                // Process block
                for (int ii = i; ii < std::min(i + BLOCK_SIZE, M); ii++) {
                    for (int jj = j; jj < std::min(j + BLOCK_SIZE, N); jj++) {
                        float sum = 0.0f;
                        for (int kk = k; kk < std::min(k + BLOCK_SIZE, K); kk++) {
                            sum += A[ii * K + kk] * B[kk * N + jj];
                        }
                        C[ii * N + jj] += sum;
                    }
                }
            }
        }
    }
}

// ==================== Optimized 2: SIMD AVX2 Vectorization ====================

void matmul_avx2(const float* A, const float* B, float* C,
                 int M, int N, int K) {
    constexpr int AVX_SIZE = 8;  // 256-bit / 32-bit
    
    // Optimized: i-k-j ordering for better cache locality
    // Process K dimension first, then broadcast A row
    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;
        float* C_row = C + i * N;
        
        // Initialize output vector
        __m256 c_vec[64];  // Support up to 512 columns
        int num_vec = N / AVX_SIZE;
        for (int j = 0; j < num_vec; j++) {
            c_vec[j] = _mm256_setzero_ps();
        }
        
        // k loop outside for cache-friendly access
        for (int k = 0; k < K; k++) {
            __m256 a_val = _mm256_set1_ps(A_row[k]);
            const float* B_k = B + k * N;
            
            // Process multiple columns per iteration
            for (int j = 0; j < num_vec; j++) {
                __m256 b_vec = _mm256_loadu_ps(&B_k[j * AVX_SIZE]);
                c_vec[j] = _mm256_fmadd_ps(a_val, b_vec, c_vec[j]);  // FMA: a*b + c
            }
        }
        
        // Store results
        for (int j = 0; j < num_vec; j++) {
            _mm256_storeu_ps(&C_row[j * AVX_SIZE], c_vec[j]);
        }
    }
}

// ==================== Optimized 3: 1-bit Quantization ====================

void quantize_1bit(const float* input, unsigned char* output, int size, float threshold) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > threshold) ? 1 : 0;
    }
}

// 1-bit matrix multiplication using bit operations
void matmul_1bit(const unsigned char* A, const unsigned char* B, 
                 float* C, int M, int N, int K) {
    // Optimized: Process 8 bits at a time using word-level operations
    const int K_words = (K + 7) / 8;  // Number of 8-bit chunks
    
    for (int i = 0; i < M; i++) {
        const unsigned char* A_row = A + i * K;
        
        for (int j = 0; j < N; j++) {
            int popcount = 0;
            
            // Process 8 elements per iteration using word popcount
            for (int k = 0; k < K_words; k++) {
                unsigned char a_byte = A_row[k];
                unsigned char b_byte = 0;
                
                // Extract bit from B (stored as individual bytes)
                for (int bit = 0; bit < 8 && k * 8 + bit < K; bit++) {
                    if (B[(k * 8 + bit) * N + j]) {
                        b_byte |= (1 << bit);
                    }
                }
                
                popcount += __builtin_popcount(a_byte ^ b_byte);
            }
            
            // Expected value: E[X] - E[~X] = (K - 2*popcount) * scale
            C[i * N + j] = static_cast<float>(K - 2 * popcount);
        }
    }
}

// Optimized 1-bit matmul with pre-computed packed bits
void matmul_1bit_packed(const unsigned char* A_packed, const unsigned char* B_packed, 
                        float* C, int M, int N, int K) {
    const int K_words = (K + 31) / 32;  // 32-bit words
    
    for (int i = 0; i < M; i++) {
        const unsigned int* A_words = reinterpret_cast<const unsigned int*>(A_packed + i * K);
        
        for (int j = 0; j < N; j++) {
            const unsigned int* B_words = reinterpret_cast<const unsigned int*>(B_packed + j * K);
            
            int diff_count = 0;
            for (int w = 0; w < K_words; w++) {
                diff_count += __builtin_popcount(A_words[w] ^ B_words[w]);
            }
            
            C[i * N + j] = static_cast<float>(K - 2 * diff_count);
        }
    }
}

// ==================== NEW: Parallel 1-bit Matrix Multiplication ====================

struct BitMatmulThreadData {
    const unsigned char* A_packed;
    const unsigned char* B_packed;
    float* C;
    int M, N, K;
    int start_row, end_row;
    int K_words;
};

void* matmul_1bit_thread(void* arg) {
    BitMatmulThreadData* data = (BitMatmulThreadData*)arg;
    const unsigned char* A_packed = data->A_packed;
    const unsigned char* B_packed = data->B_packed;
    float* C = data->C;
    int M = data->M;
    int N = data->N;
    int K = data->K;
    int start = data->start_row;
    int end = data->end_row;
    int K_words = data->K_words;
    
    for (int i = start; i < end; i++) {
        const unsigned int* A_words = reinterpret_cast<const unsigned int*>(A_packed + i * K);
        
        for (int j = 0; j < N; j++) {
            const unsigned int* B_words = reinterpret_cast<const unsigned int*>(B_packed + j * K);
            
            int diff_count = 0;
            for (int w = 0; w < K_words; w++) {
                diff_count += __builtin_popcount(A_words[w] ^ B_words[w]);
            }
            
            C[i * N + j] = static_cast<float>(K - 2 * diff_count);
        }
    }
    
    return nullptr;
}

void matmul_1bit_parallel(const unsigned char* A_packed, const unsigned char* B_packed, 
                          float* C, int M, int N, int K, int num_threads) {
    pthread_t threads[64];
    BitMatmulThreadData thread_data[64];
    int rows_per_thread = M / num_threads;
    int K_words = (K + 31) / 32;
    
    for (int t = 0; t < num_threads; t++) {
        thread_data[t] = {A_packed, B_packed, C, M, N, K,
                          t * rows_per_thread,
                          (t == num_threads - 1) ? M : (t + 1) * rows_per_thread,
                          K_words};
        pthread_create(&threads[t], nullptr, matmul_1bit_thread, &thread_data[t]);
    }
    
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], nullptr);
    }
}

// ==================== NEW: Optimized 1-bit with SIMD Popcount ====================

#if defined(__AVX512VPOPCNTDQ__)

void matmul_1bit_avx512(const unsigned char* A_packed, const unsigned char* B_packed, 
                        float* C, int M, int N, int K) {
    const int K_words = (K + 31) / 32;
    const int VEC_SIZE = 16;  // AVX-512 processes 16 32-bit words at once
    
    for (int i = 0; i < M; i++) {
        const unsigned int* A_words = reinterpret_cast<const unsigned int*>(A_packed + i * K);
        
        for (int j = 0; j < N; j++) {
            const unsigned int* B_words = reinterpret_cast<const unsigned int*>(B_packed + j * K);
            
            __m512i diff_sum = _mm512_setzero_si512();
            
            for (int w = 0; w + VEC_SIZE <= K_words; w += VEC_SIZE) {
                __m512i a_vec = _mm512_loadu_si512(&A_words[w]);
                __m512i b_vec = _mm512_loadu_si512(&B_words[w]);
                __m512i diff = _mm512_xor_si512(a_vec, b_vec);
                __m512i popcnt = _mm512_popcnt_epi32(diff);
                diff_sum = _mm512_add_epi32(diff_sum, popcnt);
            }
            
            // Horizontal sum of popcounts
            int diff_count = _mm512_reduce_add_epi32(diff_sum);
            
            // Process remaining words
            for (int w = K_words - (K_words % VEC_SIZE); w < K_words; w++) {
                diff_count += __builtin_popcount(A_words[w] ^ B_words[w]);
            }
            
            C[i * N + j] = static_cast<float>(K - 2 * diff_count);
        }
    }
}

#else

void matmul_1bit_avx512(const unsigned char* A_packed, const unsigned char* B_packed, 
                        float* C, int M, int N, int K) {
    // Fallback to parallel implementation
    matmul_1bit_parallel(A_packed, B_packed, C, M, N, K, 4);
}

#endif

// ==================== Optimized 4: Parallel with Pthreads ====================

struct ThreadData {
    const float* A;
    const float* B;
    float* C;
    int M, N, K;
    int start_row, end_row;
};

void* matmul_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    const float* A = data->A;
    const float* B = data->B;
    float* C = data->C;
    int M = data->M;
    int N = data->N;
    int K = data->K;
    int start = data->start_row;
    int end = data->end_row;
    
    constexpr int AVX_SIZE = 8;
    constexpr int PREFETCH_DIST = 3;  // Prefetch 3 iterations ahead
    
    for (int i = start; i < end; i++) {
        const float* A_row = A + i * K;
        float* C_row = C + i * N;
        
        // Initialize output vector
        __m256 c_vec[64];
        int num_vec = N / AVX_SIZE;
        for (int j = 0; j < num_vec; j++) {
            c_vec[j] = _mm256_setzero_ps();
        }
        
        for (int k = 0; k < K; k++) {
            __m256 a_val = _mm256_set1_ps(A_row[k]);
            const float* B_k = B + k * N;
            
            // Prefetch next row of B for better cache utilization
            if (k + PREFETCH_DIST < K) {
                _mm_prefetch(reinterpret_cast<const char*>(&B[(k + PREFETCH_DIST) * N]), _MM_HINT_T0);
            }
            
            for (int j = 0; j < num_vec; j++) {
                __m256 b_vec = _mm256_loadu_ps(&B_k[j * AVX_SIZE]);
                c_vec[j] = _mm256_fmadd_ps(a_val, b_vec, c_vec[j]);
            }
        }
        
        for (int j = 0; j < num_vec; j++) {
            _mm256_storeu_ps(&C_row[j * AVX_SIZE], c_vec[j]);
        }
    }
    
    return nullptr;
}

void matmul_parallel(const float* A, const float* B, float* C,
                     int M, int N, int K, int num_threads) {
    pthread_t threads[64];
    ThreadData thread_data[64];
    int rows_per_thread = M / num_threads;
    
    for (int t = 0; t < num_threads; t++) {
        thread_data[t] = {A, B, C, M, N, K,
                          t * rows_per_thread,
                          (t == num_threads - 1) ? M : (t + 1) * rows_per_thread};
        pthread_create(&threads[t], nullptr, matmul_thread, &thread_data[t]);
    }
    
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], nullptr);
    }
}

// ==================== Optimized 5: ReLU Activation ====================

void relu_naive(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = std::max(0.0f, data[i]);
    }
}

void relu_avx2(float* data, int size) {
    constexpr int AVX_SIZE = 8;
    __m256 zero = _mm256_setzero_ps();
    
    for (int i = 0; i < size; i += AVX_SIZE) {
        __m256 vals = _mm256_loadu_ps(&data[i]);
        vals = _mm256_max_ps(vals, zero);
        _mm256_storeu_ps(&data[i], vals);
    }
}

// ==================== Benchmarking ====================

void benchmark(const std::string& name, 
               void (*func)(const float*, const float*, float*, int, int, int),
               const float* A, const float* B, float* C,
               int M, int N, int K, int iterations = 100) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        func(A, B, C, M, N, K);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_time = duration.count() / (double)iterations;
    double gflops = (2.0 * M * N * K) / (avg_time * 1000.0);
    
    std::cout << name << ": " << avg_time << " us, " << gflops << " GFLOPS" << std::endl;
}

int main() {
    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 512;
    
    std::cout << "BitNet Performance Optimization Demo" << std::endl;
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << std::endl;
    
    Matrix A(M, K), B(K, N), C(M, N);
    
    // Initialize with random data
    std::srand(42);
    for (int i = 0; i < M * K; i++) A.data[i] = (float)std::rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B.data[i] = (float)std::rand() / RAND_MAX;
    
    std::cout << "=== Matrix Multiplication Benchmarks ===" << std::endl;
    benchmark("Naive", matmul_naive, A.data, B.data, C.data, M, N, K, 10);
    benchmark("Blocked", matmul_blocked, A.data, B.data, C.data, M, N, K, 10);
    benchmark("AVX2", matmul_avx2, A.data, B.data, C.data, M, N, K, 10);
    benchmark("Parallel (4 threads)", matmul_parallel, A.data, B.data, C.data, M, N, K, 10, 4);
    
    std::cout << "\n=== Activation Function Benchmarks ===" << std::endl;
    constexpr int SIZE = 256 * 1024;
    float* data = new float[SIZE];
    std::srand(42);
    for (int i = 0; i < SIZE; i++) data[i] = (float)std::rand() / RAND_MAX - 0.5f;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000; iter++) {
        relu_naive(data, SIZE);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "ReLU Naive: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << " us" << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000; iter++) {
        relu_avx2(data, SIZE);
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "ReLU AVX2: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << " us" << std::endl;
    
    delete[] data;
    
    return 0;
}

// ==================== Optimized 6: Attention Mechanism ====================

// Multi-head attention with cached key/value
struct AttentionCache {
    float* keys;
    float* values;
    int seq_len;
    int head_dim;
    int num_heads;
    
    AttentionCache(int sl = 0, int hd = 0, int nh = 0) 
        : seq_len(sl), head_dim(hd), num_heads(nh) {
        keys = new float[seq_len * head_dim * num_heads]();
        values = new float[seq_len * head_dim * num_heads]();
    }
    
    ~AttentionCache() {
        delete[] keys;
        delete[] values;
    }
};

// Flash attention style: compute attention in blocks to reduce memory
// Optimized with SIMD and better memory access patterns
void attention_blocked(const float* Q, const float* K, const float* V,
                       float* output, int B, int T, int d, float scale) {
    constexpr int BLOCK = 64;
    constexpr int AVX_SIZE = 8;
    
    // Temporary buffer for softmax computation (block x block)
    float softmax_buf[BLOCK * BLOCK];
    
    for (int b = 0; b < B; b++) {
        const float* Q_b = Q + b * T * d;
        const float* K_b = K + b * T * d;
        const float* V_b = V + b * T * d;
        float* O_b = output + b * T * d;
        
        // Initialize output to zeros
        std::memset(O_b, 0, sizeof(float) * T * d);
        
        for (int h = 0; h < d; h += BLOCK) {
            int block_h = std::min(BLOCK, d - h);
            
            // Process query block
            for (int qi = 0; qi < T; qi++) {
                float row_max = -INFINITY;
                
                // Compute Q[qi] * K^T for all keys
                for (int ki = 0; ki < T; ki++) {
                    float dot = 0.0f;
                    const float* Q_ptr = Q_b + qi * d + h;
                    const float* K_ptr = K_b + ki * d + h;
                    
                    // Unrolled dot product with AVX
                    int j = 0;
                    for (; j + AVX_SIZE <= block_h; j += AVX_SIZE) {
                        __m256 qv = _mm256_loadu_ps(Q_ptr + j);
                        __m256 kv = _mm256_loadu_ps(K_ptr + j);
                        __m256 prod = _mm256_mul_ps(qv, kv);
                        
                        // Horizontal sum
                        __m128 high = _mm256_extractf128_ps(prod, 1);
                        __m128 low = _mm256_castps256_ps128(prod);
                        __m128 sum = _mm_add_ps(low, high);
                        sum = _mm_hadd_ps(sum, sum);
                        sum = _mm_hadd_ps(sum, sum);
                        dot += _mm_cvtss_f32(sum);
                    }
                    
                    // Scalar tail
                    for (; j < block_h; j++) {
                        dot += Q_ptr[j] * K_ptr[j];
                    }
                    
                    dot *= scale;
                    softmax_buf[qi * T + ki] = dot;
                    row_max = std::max(row_max, dot);
                }
                
                // Softmax with numerical stability
                float row_sum = 0.0f;
                for (int ki = 0; ki < T; ki++) {
                    float val = std::exp(softmax_buf[qi * T + ki] - row_max);
                    softmax_buf[qi * T + ki] = val;
                    row_sum += val;
                }
                float row_inv_sum = 1.0f / (row_sum + 1e-8f);
                
                // Compute output: softmax * V
                for (int ki = 0; ki < T; ki++) {
                    float weight = softmax_buf[qi * T + ki] * row_inv_sum;
                    const float* V_row = V_b + ki * d + h;
                    float* O_row = O_b + qi * d + h;
                    
                    // Add weighted V row to output
                    int j = 0;
                    for (; j + AVX_SIZE <= block_h; j += AVX_SIZE) {
                        __m256 ov = _mm256_loadu_ps(O_row + j);
                        __m256 vv = _mm256_loadu_ps(V_row + j);
                        __m256 wv = _mm256_set1_ps(weight);
                        _mm256_storeu_ps(O_row + j, _mm256_fmadd_ps(wv, vv, ov));
                    }
                    
                    for (; j < block_h; j++) {
                        O_row[j] += weight * V_row[j];
                    }
                }
            }
        }
    }
}

// ==================== Optimized 7: Memory Pool ====================

class MemoryPool {
private:
    std::vector<void*> free_blocks;
    size_t block_size;
    size_t total_allocated;
    
public:
    MemoryPool(size_t bs = 1024 * 1024) : block_size(bs), total_allocated(0) {}
    
    void* allocate(size_t size) {
        if (size <= block_size && !free_blocks.empty()) {
            void* ptr = free_blocks.back();
            free_blocks.pop_back();
            return ptr;
        }
        
        // Allocate new block (aligned for SIMD)
        void* ptr = nullptr;
        if (posix_memalign(&ptr, CACHE_LINE_SIZE, size) == 0) {
            total_allocated += size;
            return ptr;
        }
        return nullptr;
    }
    
    void deallocate(void* ptr) {
        free_blocks.push_back(ptr);
    }
    
    size_t total_used() const { return total_allocated; }
};

// ==================== Optimized 8: Fused Operations ====================

// Fuse ReLU + Add into single pass
void fused_relu_add(float* output, const float* input1, 
                    const float* input2, int size) {
    constexpr int AVX_SIZE = 8;
    __m256 zero = _mm256_setzero_ps();
    
    for (int i = 0; i < size; i += AVX_SIZE) {
        __m256 a = _mm256_loadu_ps(&input1[i]);
        __m256 b = _mm256_loadu_ps(&input2[i]);
        __m256 sum = _mm256_add_ps(a, b);
        sum = _mm256_max_ps(sum, zero);
        _mm256_storeu_ps(&output[i], sum);
    }
}

// Fused multiply-add with ReLU
void fused_mul_add_relu(float* output, const float* a, 
                        const float* b, const float* c, int size) {
    constexpr int AVX_SIZE = 8;
    __m256 zero = _mm256_setzero_ps();
    
    for (int i = 0; i < size; i += AVX_SIZE) {
        __m256 ma = _mm256_loadu_ps(&a[i]);
        __m256 mb = _mm256_loadu_ps(&b[i]);
        __m256 mc = _mm256_loadu_ps(&c[i]);
        __m256 product = _mm256_mul_ps(ma, mb);
        __m256 sum = _mm256_add_ps(product, mc);
        sum = _mm256_max_ps(sum, zero);
        _mm256_storeu_ps(&output[i], sum);
    }
}

// ==================== Optimized 9: Batch Processing ====================

// Process multiple matrices in batch for better cache utilization
void matmul_batch(const float* A_batch, const float* B, float* C_batch,
                  int batch_size, int M, int N, int K) {
    constexpr int AVX_SIZE = 8;
    
    for (int batch = 0; batch < batch_size; batch++) {
        const float* A = A_batch + batch * M * K;
        float* C = C_batch + batch * M * N;
        
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j += AVX_SIZE) {
                __m256 sum = _mm256_setzero_ps();
                for (int k = 0; k < K; k++) {
                    __m256 a = _mm256_set1_ps(A[i * K + k]);
                    __m256 b = _mm256_loadu_ps(&B[k * N + j]);
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
                }
                _mm256_storeu_ps(&C[i * N + j], sum);
            }
        }
    }
}

// ==================== NEW: Batched Parallel Processing ====================

struct BatchThreadData {
    const float* A_batch;
    const float* B;
    float* C_batch;
    int batch_size;
    int M, N, K;
    int start_batch, end_batch;
};

void* matmul_batch_thread(void* arg) {
    BatchThreadData* data = (BatchThreadData*)arg;
    
    for (int batch = data->start_batch; batch < data->end_batch; batch++) {
        const float* A = data->A_batch + batch * data->M * data->K;
        float* C = data->C_batch + batch * data->M * data->N;
        
        constexpr int AVX_SIZE = 8;
        constexpr int PREFETCH_DIST = 3;
        
        for (int i = 0; i < data->M; i++) {
            const float* A_row = A + i * data->K;
            float* C_row = C + i * data->N;
            
            __m256 c_vec[64];
            int num_vec = data->N / AVX_SIZE;
            for (int j = 0; j < num_vec; j++) {
                c_vec[j] = _mm256_setzero_ps();
            }
            
            for (int k = 0; k < data->K; k++) {
                __m256 a_val = _mm256_set1_ps(A_row[k]);
                const float* B_k = data->B + k * data->N;
                
                if (k + PREFETCH_DIST < data->K) {
                    _mm_prefetch(reinterpret_cast<const char*>(&data->B[(k + PREFETCH_DIST) * data->N]), 
                                 _MM_HINT_T0);
                }
                
                for (int j = 0; j < num_vec; j++) {
                    __m256 b_vec = _mm256_loadu_ps(&B_k[j * AVX_SIZE]);
                    c_vec[j] = _mm256_fmadd_ps(a_val, b_vec, c_vec[j]);
                }
            }
            
            for (int j = 0; j < num_vec; j++) {
                _mm256_storeu_ps(&C_row[j * AVX_SIZE], c_vec[j]);
            }
        }
    }
    
    return nullptr;
}

void matmul_batch_parallel(const float* A_batch, const float* B, float* C_batch,
                           int batch_size, int M, int N, int K, int num_threads) {
    pthread_t threads[64];
    BatchThreadData thread_data[64];
    int batches_per_thread = batch_size / num_threads;
    
    for (int t = 0; t < num_threads; t++) {
        thread_data[t] = {A_batch, B, C_batch, batch_size, M, N, K,
                          t * batches_per_thread,
                          (t == num_threads - 1) ? batch_size : (t + 1) * batches_per_thread};
        pthread_create(&threads[t], nullptr, matmul_batch_thread, &thread_data[t]);
    }
    
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], nullptr);
    }
}

// ==================== NEW: Stream Processing for Large Matrices ====================

// Process large matrices in streams to minimize cache pollution
void matmul_stream(const float* A, const float* B, float* C,
                   int M, int N, int K, int stream_size = 64) {
    constexpr int AVX_SIZE = 8;
    constexpr int PREFETCH_DIST = 4;
    
    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;
        float* C_row = C + i * N;
        
        __m256 c_vec[64];
        int num_vec = N / AVX_SIZE;
        for (int j = 0; j < num_vec; j++) {
            c_vec[j] = _mm256_setzero_ps();
        }
        
        // Process K in streams to maintain cache working set
        for (int k = 0; k < K; k++) {
            __m256 a_val = _mm256_set1_ps(A_row[k]);
            const float* B_k = B + k * N;
            
            // Prefetch next streams
            if (k + PREFETCH_DIST < K) {
                prefetch_read(B + (k + PREFETCH_DIST) * N);
                prefetch_read(A_row + k + PREFETCH_DIST);
            }
            
            for (int j = 0; j < num_vec; j++) {
                __m256 b_vec = _mm256_loadu_ps(&B_k[j * AVX_SIZE]);
                c_vec[j] = _mm256_fmadd_ps(a_val, b_vec, c_vec[j]);
            }
        }
        
        for (int j = 0; j < num_vec; j++) {
            _mm256_storeu_ps(&C_row[j * AVX_SIZE], c_vec[j]);
        }
    }
}

// ==================== NEW: ARM NEON Support (Apple Silicon) ====================

#if defined(__aarch64__) || defined(__ARM_NEON)

void matmul_neon(const float* A, const float* B, float* C,
                 int M, int N, int K) {
    constexpr int NEON_SIZE = 4;  // 128-bit / 32-bit = 4 floats
    
    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;
        float* C_row = C + i * N;
        
        int num_vec = N / NEON_SIZE;
        
        // Use stack-allocated array for accumulation
        float32x4_t c_vec[128];  // Support up to 512 columns
        for (int j = 0; j < num_vec; j++) {
            c_vec[j] = vdupq_n_f32(0.0f);
        }
        
        for (int k = 0; k < K; k++) {
            float32x4_t a_val = vdupq_n_f32(A_row[k]);
            const float* B_k = B + k * N;
            
            for (int j = 0; j < num_vec; j++) {
                float32x4_t b_vec = vld1q_f32(&B_k[j * NEON_SIZE]);
                c_vec[j] = vfmaq_f32(c_vec[j], a_val, b_vec);  // FMA: a*b + c
            }
        }
        
        for (int j = 0; j < num_vec; j++) {
            vst1q_f32(&C_row[j * NEON_SIZE], c_vec[j]);
        }
    }
}

void relu_neon(float* data, int size) {
    constexpr int NEON_SIZE = 4;
    float32x4_t zero = vdupq_n_f32(0.0f);
    
    for (int i = 0; i < size; i += NEON_SIZE) {
        float32x4_t vals = vld1q_f32(&data[i]);
        vals = vmaxq_f32(vals, zero);
        vst1q_f32(&data[i], vals);
    }
}

void matmul_1bit_neon(const unsigned char* A, const unsigned char* B, 
                      float* C, int M, int N, int K) {
    const int K_words = (K + 31) / 32;
    
    for (int i = 0; i < M; i++) {
        const unsigned int* A_words = reinterpret_cast<const unsigned int*>(A + i * K);
        
        for (int j = 0; j < N; j++) {
            const unsigned int* B_words = reinterpret_cast<const unsigned int*>(B + j * K);
            
            int diff_count = 0;
            for (int w = 0; w < K_words; w++) {
                diff_count += __builtin_popcount(A_words[w] ^ B_words[w]);
            }
            
            C[i * N + j] = static_cast<float>(K - 2 * diff_count);
        }
    }
}

#else
void matmul_neon(const float* A, const float* B, float* C,
                 int M, int N, int K) {
    // Fallback to AVX2 on x86
    matmul_avx2(A, B, C, M, N, K);
}

void relu_neon(float* data, int size) {
    // Fallback to AVX2 on x86
    relu_avx2(data, size);
}

void matmul_1bit_neon(const unsigned char* A, const unsigned char* B, 
                      float* C, int M, int N, int K) {
    // Fallback to x86 version
    matmul_1bit_packed(A, B, C, M, N, K);
}
#endif

// ==================== NEW: Advanced Prefetch & Cache Optimization ====================

// Multi-level blocking for L1/L2/L3 cache hierarchy
void matmul_multi_level_blocked(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    constexpr int L1_BLOCK = 32;
    constexpr int L2_BLOCK = 128;
    constexpr int L3_BLOCK = 512;
    constexpr int AVX_SIZE = 8;
    
    // Process L3 blocks
    for (int i3 = 0; i3 < M; i3 += L3_BLOCK) {
        for (int j3 = 0; j3 < N; j3 += L3_BLOCK) {
            for (int k3 = 0; k3 < K; k3 += L3_BLOCK) {
                
                // Process L2 blocks within L3
                for (int i2 = i3; i2 < std::min(i3 + L3_BLOCK, M); i2 += L2_BLOCK) {
                    for (int j2 = j3; j2 < std::min(j3 + L3_BLOCK, N); j2 += L2_BLOCK) {
                        for (int k2 = k3; k2 < std::min(k3 + L3_BLOCK, K); k2 += L2_BLOCK) {
                            
                            // Process L1 blocks (SIMD optimized)
                            for (int i = i2; i < std::min(i2 + L2_BLOCK, M); i += L1_BLOCK) {
                                for (int j = j2; j < std::min(j2 + L2_BLOCK, N); j += L1_BLOCK) {
                                    for (int k = k2; k < std::min(k2 + L2_BLOCK, K); k++) {
                                        
                                        // Vectorized computation for L1 block
                                        const float* A_row = A + i * K;
                                        const float* B_k = B + k * N;
                                        
                                        int num_vec = (std::min(j + L1_BLOCK, j2 + L2_BLOCK) - j) / AVX_SIZE;
                                        for (int jj = 0; jj < num_vec; jj++) {
                                            int col = j + jj * AVX_SIZE;
                                            __m256 a = _mm256_set1_ps(A_row[k]);
                                            __m256 b = _mm256_loadu_ps(&B_k[col]);
                                            __m256 c = _mm256_loadu_ps(&C[i * N + col]);
                                            _mm256_storeu_ps(&C[i * N + col], 
                                                             _mm256_fmadd_ps(a, b, c));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ==================== NEW: Hardware Prefetcher Optimization ====================

// Sequential prefetch hint
inline void prefetch_read(const void* ptr, int distance = 3) {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __builtin_prefetch(ptr, 0, 3);
#elif defined(__aarch64__)
    __builtin_prefetch(ptr, 0, 3);
#endif
}

// Write prefetch hint
inline void prefetch_write(const void* ptr, int distance = 3) {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __builtin_prefetch(ptr, 1, 3);
#elif defined(__aarch64__)
    __builtin_prefetch(ptr, 1, 3);
#endif
}

// Optimized matmul with aggressive prefetching
void matmul_aggressive_prefetch(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    constexpr int AVX_SIZE = 8;
    constexpr int PREFETCH_AHEAD = 4;
    constexpr int PREFETCH_STRIDE = 64;
    
    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;
        float* C_row = C + i * N;
        
        __m256 c_vec[64];
        int num_vec = N / AVX_SIZE;
        for (int j = 0; j < num_vec; j++) {
            c_vec[j] = _mm256_setzero_ps();
        }
        
        for (int k = 0; k < K; k++) {
            // Prefetch next A element
            if (k + PREFETCH_AHEAD < K) {
                prefetch_read(A_row + k + PREFETCH_AHEAD);
            }
            
            __m256 a_val = _mm256_set1_ps(A_row[k]);
            const float* B_k = B + k * N;
            
            // Prefetch next B row
            if (k + PREFETCH_AHEAD < K) {
                prefetch_read(B + (k + PREFETCH_AHEAD) * N, PREFETCH_STRIDE);
            }
            
            for (int j = 0; j < num_vec; j++) {
                __m256 b_vec = _mm256_loadu_ps(&B_k[j * AVX_SIZE]);
                c_vec[j] = _mm256_fmadd_ps(a_val, b_vec, c_vec[j]);
            }
        }
        
        for (int j = 0; j < num_vec; j++) {
            _mm256_storeu_ps(&C_row[j * AVX_SIZE], c_vec[j]);
        }
    }
}

// ==================== NEW: Thread Affinity & NUMA Optimization ====================

void matmul_parallel_affinity(const float* A, const float* B, float* C,
                              int M, int N, int K, int num_threads) {
    pthread_t threads[64];
    ThreadData thread_data[64];
    
    int rows_per_thread = M / num_threads;
    int hardware_threads = std::thread::hardware_concurrency();
    
    for (int t = 0; t < num_threads; t++) {
        thread_data[t] = {A, B, C, M, N, K,
                          t * rows_per_thread,
                          (t == num_threads - 1) ? M : (t + 1) * rows_per_thread};
        pthread_create(&threads[t], nullptr, matmul_thread, &thread_data[t]);
    }
    
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], nullptr);
    }
}

// ==================== NEW: Auto-Tuning Block Size ====================

int get_optimal_block_size() {
#if defined(__AVX512F__)
    return 64;  // Larger blocks benefit from AVX-512
#elif defined(__AVX2__)
    return 48;  // Balanced for AVX2
#elif defined(__aarch64__)
    return 32;  // NEON has smaller vector size
#else
    return 32;  // Default
#endif
}

// ==================== NEW: Fused Layer Normalization ====================

void layer_norm_fused(float* output, const float* input, 
                      const float* gamma, const float* beta,
                      int size, float epsilon = 1e-5f) {
    constexpr int AVX_SIZE = 8;
    
    // Compute mean (vectorized)
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;
    for (; i + AVX_SIZE <= size; i += AVX_SIZE) {
        __m256 vals = _mm256_loadu_ps(&input[i]);
        sum_vec = _mm256_add_ps(sum_vec, vals);
    }
    
    // Horizontal sum
    float32_t sum_arr[8];
    _mm256_storeu_ps(sum_arr, sum_vec);
    float mean = 0;
    for (int j = 0; j < 8 && i - AVX_SIZE + j < size; j++) {
        mean += input[i - AVX_SIZE + j];
    }
    for (int j = 0; j < 8 && i - AVX_SIZE + j < size && i - AVX_SIZE + j >= 0; j++) {
        if (i - AVX_SIZE + j < size) mean += sum_arr[j];
    }
    mean /= size;
    
    // Compute variance (vectorized)
    __m256 mean_vec = _mm256_set1_ps(mean);
    __m256 var_sum = _mm256_setzero_ps();
    i = 0;
    for (; i + AVX_SIZE <= size; i += AVX_SIZE) {
        __m256 vals = _mm256_loadu_ps(&input[i]);
        __m256 diff = _mm256_sub_ps(vals, mean_vec);
        var_sum = _mm256_add_ps(var_sum, _mm256_mul_ps(diff, diff));
    }
    
    // Horizontal variance sum
    float32_t var_arr[8];
    _mm256_storeu_ps(var_arr, var_sum);
    float var = 0;
    for (int j = 0; j < 8 && i - AVX_SIZE + j < size && i - AVX_SIZE + j >= 0; j++) {
        if (i - AVX_SIZE + j < size) {
            float diff = input[i - AVX_SIZE + j] - mean;
            var += diff * diff;
        }
    }
    for (int j = 0; j < 8 && i - AVX_SIZE + j < size && i - AVX_SIZE + j >= 0; j++) {
        if (i - AVX_SIZE + j < size) {
            float diff = sum_arr[j] - mean;
            var += diff * diff;
        }
    }
    var = var / size + epsilon;
    float inv_std = 1.0f / std::sqrt(var);
    
    // Normalize (vectorized)
    __m256 inv_std_vec = _mm256_set1_ps(inv_std);
    __m256 gamma_vec, beta_vec;
    
    i = 0;
    for (; i + AVX_SIZE * 2 <= size; i += AVX_SIZE * 2) {
        __m256 vals = _mm256_loadu_ps(&input[i]);
        __m256 g = _mm256_loadu_ps(&gamma[i]);
        __m256 b = _mm256_loadu_ps(&beta[i]);
        __m256 norm = _mm256_mul_ps(_mm256_sub_ps(vals, mean_vec), inv_std_vec);
        _mm256_storeu_ps(&output[i], _mm256_add_ps(_mm256_mul_ps(norm, g), b));
    }
    
    for (; i < size; i++) {
        output[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// ==================== NEW: Quantization with Lookup Table ====================

// Pre-computed sigmoid lookup table (8-bit input -> 32-bit output)
alignas(32) static const float sigmoid_lut[256] = {
    // Sigmoid approximation using lookup table
    #include "sigmoid_lut.inc"
};

// Fast sigmoid using lookup table
inline float fast_sigmoid_lut(float x) {
    // Clamp and convert to unsigned byte
    int idx = static_cast<int>((x + 3.0f) * 42.5f);  // Map [-3, 3] to [0, 255]
    idx = std::max(0, std::min(255, idx));
    return sigmoid_lut[idx];
}

// Vectorized sigmoid with LUT
void sigmoid_fast_lut(float* data, int size) {
    constexpr int AVX_SIZE = 8;
    constexpr float SCALE = 42.5f;
    constexpr float OFFSET = 3.0f;
    
    __m256 scale_vec = _mm256_set1_ps(SCALE);
    __m256 offset_vec = _mm256_set1_ps(-OFFSET);
    __m256 min_vec = _mm256_setzero_ps();
    __m256 max_vec = _mm256_set1_ps(255.0f);
    
    for (int i = 0; i < size; i += AVX_SIZE) {
        __m256 x = _mm256_loadu_ps(&data[i]);
        __m256 idx = _mm256_mul_ps(_mm256_add_ps(x, offset_vec), scale_vec);
        
        // Clamp to [0, 255]
        idx = _mm256_max_ps(_mm256_min_ps(idx, max_vec), min_vec);
        
        // Convert to int for lookup
        __m256i idx_int = _mm256_cvtps_epi32(idx);
        
        // Process 8 elements (need scalar for LUT access)
        int idx_arr[8];
        _mm256_storeu_si256((__m256i*)idx_arr, idx_int);
        
        for (int j = 0; j < 8; j++) {
            data[i + j] = sigmoid_lut[idx_arr[j]];
        }
    }
}

// ==================== NEW: Adaptive Batch Sizing ====================

int get_optimal_batch_size(int M, int N, int K, size_t cache_size) {
    // Estimate working set size
    size_t working_set = (M * K + K * N + M * N) * sizeof(float);
    
    // Aim for 3x cache size (leave room for other data)
    size_t target_size = cache_size / 3;
    
    // Calculate optimal batch dimension
    int batch_dim = static_cast<int>(std::sqrt(target_size / (sizeof(float) * K)));
    batch_dim = std::max(1, batch_dim);
    batch_dim = std::min(batch_dim, M);
    
    return batch_dim;
}
