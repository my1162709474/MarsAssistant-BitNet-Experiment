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
#include <pthread.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

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
        data = new float[rows * cols];
        std::memset(data, 0, sizeof(float) * rows * cols);
    }
    
    ~Matrix() {
        delete[] data;
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
