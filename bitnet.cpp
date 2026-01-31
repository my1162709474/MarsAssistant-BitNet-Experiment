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

// ==================== Optimized 3: 1-bit Quantization ====================

void quantize_1bit(const float* input, unsigned char* output, int size, float threshold) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > threshold) ? 1 : 0;
    }
}

// 1-bit matrix multiplication using bit operations
void matmul_1bit(const unsigned char* A, const unsigned char* B, 
                 float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int popcount = 0;
            for (int k = 0; k < K; k++) {
                // XOR gives 1 where bits differ
                popcount += __builtin_popcount(A[i * K + k] ^ B[k * N + j]);
            }
            // Expected value: E[X] - E[~X] = (K - 2*popcount) * scale
            C[i * N + j] = static_cast<float>(K - 2 * popcount);
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
    
    for (int i = start; i < end; i++) {
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
void attention_blocked(const float* Q, const float* K, const float* V,
                       float* output, int B, int T, int d, float scale) {
    constexpr int BLOCK = 64;
    
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < d; h += BLOCK) {
            // Process in blocks to stay in L1 cache
            float block_max = -INFINITY;
            float block_sum = 0.0f;
            
            // Compute Q*K^T in blocks
            for (int t = 0; t < T; t += BLOCK) {
                for (int i = 0; i < BLOCK; i++) {
                    if (t + i >= T) break;
                    
                    float qk = 0.0f;
                    for (int j = 0; j < BLOCK; j++) {
                        if (h + j >= d) break;
                        qk += Q[b * T * d + (t + i) * d + (h + j)] * 
                              K[b * T * d + (t + j) * d + (h + j)];
                    }
                    qk *= scale;
                    
                    // Softmax with numerical stability
                    qk = std::exp(qk - block_max);
                    block_sum += qk;
                    
                    // Store intermediate
                    float* temp = output + b * T * d + (t + i) * d + h;
                    temp[0] = qk;  // Using first element as temp storage
                }
            }
            
            // Normalize
            float inv_sum = 1.0f / (block_sum + 1e-8f);
            
            // Compute output: softmax(Q*K^T) * V
            for (int i = 0; i < BLOCK; i++) {
                if (h + i >= d) break;
                float attn_weight = output[b * T * d + h * d + i] * inv_sum;  // Simplified
                for (int t = 0; t < T; t++) {
                    output[b * T * d + t * d + (h + i)] += 
                        attn_weight * V[b * T * d + t * d + (h + i)];
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
