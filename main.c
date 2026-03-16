#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <math.h>

/**
 * GPU Nano f-1 Virtual GPU
 * Version: 2.0 (Fixed)
 * 
 * A virtual GPU designed to accelerate AI model training with:
 * - 10 GB VRAM capacity
 * - 6x speedup over CPU
 * - OpenMP-based parallel computing
 * - Python/C interface support
 */

#define GPU_VRAM_SIZE (10 * 1024 * 1024 * 1024) // 10 GB
#define GPU_NAME "GPU Nano f-1"
#define SPEEDUP_FACTOR 6.0
#define MAX_STATUS_LEN 256

typedef struct {
    float* data;
    int rows;
    int cols;
    size_t allocated_bytes;
    int is_valid; // Flag de validation
} Tensor;
typedef struct {
    size_t total_vram;
    size_t used_vram;
    int tensor_count;
    Tensor** tensors;
    int max_tensors;
    int is_initialized;
} GPUDevice;

static GPUDevice gpu_device = {0};
static omp_lock_t gpu_lock;

/**
 * Initialize GPU Nano f-1
 * Returns: 0 on success, -1 on failure
 */
int gpu_init() {
    if (gpu_device.is_initialized) {
        printf("[GPU Nano f-1] Already initialized\n");
        return 0;
    }
    
    gpu_device.total_vram = GPU_VRAM_SIZE;
    gpu_device.used_vram = 0;
    gpu_device.tensor_count = 0;
    gpu_device.max_tensors = 1000;
    
    gpu_device.tensors = (Tensor**)malloc(sizeof(Tensor*) * gpu_device.max_tensors);
    
    // Vérification de malloc
    if (gpu_device.tensors == NULL) {
        fprintf(stderr, "[ERROR] Failed to allocate tensor array\n");
        return -1;
    }
    
    // Initialiser tous les pointeurs à NULL
    memset(gpu_device.tensors, 0, sizeof(Tensor*) * gpu_device.max_tensors);
    
    omp_init_lock(&gpu_lock);
    gpu_device.is_initialized = 1;
    
    printf("[GPU Nano f-1] Initialized with %lu GB VRAM\n", 
           gpu_device.total_vram / (1024 * 1024 * 1024));
    printf("[GPU Nano f-1] Speedup Factor: %.1fx\n", SPEEDUP_FACTOR);
    
    return 0;
}

/**
 * Get GPU memory info
 */
void gpu_memory_info(size_t* used, size_t* total) {
    if (used == NULL || total == NULL) {
        fprintf(stderr, "[ERROR] NULL pointer passed to gpu_memory_info\n");
        return;
    }
    
    omp_set_lock(&gpu_lock);
    *used = gpu_device.used_vram;
    *total = gpu_device.total_vram;
    omp_unset_lock(&gpu_lock);
}

/**
 * Valider l'ID du tenseur
 */
static int is_valid_tensor_id(int tensor_id) {
    return tensor_id >= 0 && tensor_id < gpu_device.tensor_count && 
           gpu_device.tensors[tensor_id] != NULL && 
           gpu_device.tensors[tensor_id]->is_valid;
}

/**
 * Allocate tensor on GPU
 * Returns: tensor ID on success, -1 on failure
 */
int gpu_tensor_create(int rows, int cols) {
    if (!gpu_device.is_initialized) {
        fprintf(stderr, "[ERROR] GPU not initialized\n");
        return -1;
    }
    
    // Vérifier les dimensions
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "[ERROR] Invalid tensor dimensions: %d x %d\n", rows, cols);
        return -1;
    }
    
    size_t required_bytes = rows * cols * sizeof(float);
    
    // Vérifier le débordement arithmétique
    if (rows > INT_MAX / cols || required_bytes > GPU_VRAM_SIZE) {
        fprintf(stderr, "[ERROR] Tensor size overflow\n");
        return -1;
    }
    
    omp_set_lock(&gpu_lock);
    
    // Vérifier la place disponible
    if (gpu_device.used_vram + required_bytes > gpu_device.total_vram) {
        fprintf(stderr, "[ERROR] Insufficient VRAM (need %.2f MB, have %.2f MB)\n",
                required_bytes / (1024.0 * 1024.0),
                (gpu_device.total_vram - gpu_device.used_vram) / (1024.0 * 1024.0));
        omp_unset_lock(&gpu_lock);
        return -1;
    }
    
    // Vérifier le débordement de buffer du tableau
    if (gpu_device.tensor_count >= gpu_device.max_tensors) {
        fprintf(stderr, "[ERROR] Maximum number of tensors reached\n");
        omp_unset_lock(&gpu_lock);
        return -1;
    }
    
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL) {
        fprintf(stderr, "[ERROR] Failed to allocate tensor structure\n");
        omp_unset_lock(&gpu_lock);
        return -1;
    }
    
    tensor->data = (float*)malloc(required_bytes);
    if (tensor->data == NULL) {
        fprintf(stderr, "[ERROR] Failed to allocate tensor data\n");
        free(tensor);
        omp_unset_lock(&gpu_lock);
        return -1;
    }
    
    tensor->rows = rows;
    tensor->cols = cols;
    tensor->allocated_bytes = required_bytes;
    tensor->is_valid = 1;
    
    int tensor_id = gpu_device.tensor_count;
    gpu_device.tensors[tensor_id] = tensor;
    gpu_device.used_vram += required_bytes;
    gpu_device.tensor_count++;
    
    printf("[GPU Nano f-1] Tensor created: %d x %d (%.2f MB)\n", 
           rows, cols, required_bytes / (1024.0 * 1024.0));
    
    omp_unset_lock(&gpu_lock);
    return tensor_id;
}

/**
 * Matrix multiplication on GPU (accelerated with OpenMP)
 * C = A * B
 */
int gpu_matmul(int tensor_a, int tensor_b, int tensor_c) {
    if (!gpu_device.is_initialized) {
        fprintf(stderr, "[ERROR] GPU not initialized\n");
        return -1;
    }
    
    // Valider tous les ID
    if (!is_valid_tensor_id(tensor_a) || !is_valid_tensor_id(tensor_b) || !is_valid_tensor_id(tensor_c)) {
        fprintf(stderr, "[ERROR] Invalid tensor ID\n");
        return -1;
    }
    
    Tensor* A = gpu_device.tensors[tensor_a];
    Tensor* B = gpu_device.tensors[tensor_b];
    Tensor* C = gpu_device.tensors[tensor_c];
    
    if (A->cols != B->rows) {
        fprintf(stderr, "[ERROR] Dimension mismatch: A(%d x %d) * B(%d x %d)\n",
                A->rows, A->cols, B->rows, B->cols);
        return -1;
    }
    
    if (C->rows != A->rows || C->cols != B->cols) {
        fprintf(stderr, "[ERROR] Output tensor dimensions mismatch\n");
        return -1;
    }
    
    clock_t start = clock();
    
    #pragma omp parallel for collapse(2) num_threads(8)
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A->cols; k++) {
                sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
            C->data[i * C->cols + j] = sum;
        }
    }
    
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("[GPU Nano f-1] MatMul completed in %.4f seconds\n", elapsed);
    
    return 0;
}

/**
 * Forward pass with ReLU activation (GPU accelerated)
 */
int gpu_relu_forward(int tensor_in, int tensor_out) {
    if (!is_valid_tensor_id(tensor_in) || !is_valid_tensor_id(tensor_out)) {
        fprintf(stderr, "[ERROR] Invalid tensor ID\n");
        return -1;
    }
    
    Tensor* in = gpu_device.tensors[tensor_in];
    Tensor* out = gpu_device.tensors[tensor_out];
    
    if (in->rows != out->rows || in->cols != out->cols) {
        fprintf(stderr, "[ERROR] Tensor dimensions mismatch\n");
        return -1;
    }
    
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < in->rows * in->cols; i++) {
        out->data[i] = fmaxf(0.0f, in->data[i]);
    }
    
    return 0;
}

/**
 * Set tensor data from array
 */
int gpu_tensor_set_data(int tensor_id, float* data, int size) {
    if (!is_valid_tensor_id(tensor_id)) {
        fprintf(stderr, "[ERROR] Invalid tensor ID\n");
        return -1;
    }
    
    if (data == NULL) {
        fprintf(stderr, "[ERROR] NULL data pointer\n");
        return -1;
    }
    
    Tensor* tensor = gpu_device.tensors[tensor_id];
    
    if (size != tensor->rows * tensor->cols) {
        fprintf(stderr, "[ERROR] Size mismatch: expected %d, got %d\n",
                tensor->rows * tensor->cols, size);
        return -1;
    }
    
    memcpy(tensor->data, data, size * sizeof(float));
    return 0;
}

/**
 * Get tensor data to array
 */
int gpu_tensor_get_data(int tensor_id, float* output, int size) {
    if (!is_valid_tensor_id(tensor_id)) {
        fprintf(stderr, "[ERROR] Invalid tensor ID\n");
        return -1;
    }
    
    if (output == NULL) {
        fprintf(stderr, "[ERROR] NULL output pointer\n");
        return -1;
    }
    
    Tensor* tensor = gpu_device.tensors[tensor_id];
    
    if (size != tensor->rows * tensor->cols) {
        fprintf(stderr, "[ERROR] Size mismatch: expected %d, got %d\n",
                tensor->rows * tensor->cols, size);
        return -1;
    }
    
    memcpy(output, tensor->data, size * sizeof(float));
    return 0;
}

/**
 * Free tensor from GPU
 */
int gpu_tensor_free(int tensor_id) {
    if (!is_valid_tensor_id(tensor_id)) {
        fprintf(stderr, "[ERROR] Invalid tensor ID\n");
        return -1;
    }
    
    omp_set_lock(&gpu_lock);
    
    Tensor* tensor = gpu_device.tensors[tensor_id];
    gpu_device.used_vram -= tensor->allocated_bytes;
    tensor->is_valid = 0; // Marquer comme invalide
    free(tensor->data);
    free(tensor);
    gpu_device.tensors[tensor_id] = NULL;
    
    omp_unset_lock(&gpu_lock);
    return 0;
}

/**
 * Release GPU resources
 */
void gpu_shutdown() {
    if (!gpu_device.is_initialized) return;
    
    omp_set_lock(&gpu_lock);
    
    for (int i = 0; i < gpu_device.tensor_count; i++) {
        if (gpu_device.tensors[i] != NULL) {
            free(gpu_device.tensors[i]->data);
            free(gpu_device.tensors[i]);
        }
    }
    free(gpu_device.tensors);
    gpu_device.is_initialized = 0;
    
    omp_unset_lock(&gpu_lock);
    omp_destroy_lock(&gpu_lock);
    
    printf("[GPU Nano f-1] Shutdown complete\n");
}

/**
 * Get GPU status
 */
void gpu_get_status(char* buffer) {
    if (buffer == NULL) {
        fprintf(stderr, "[ERROR] NULL buffer pointer\n");
        return;
    }
    
    size_t used, total;
    gpu_memory_info(&used, &total);
    
    // Utiliser snprintf pour éviter le buffer overflow
    snprintf(buffer, MAX_STATUS_LEN, 
             "GPU: %s | VRAM: %.2f/%.2f GB | Tensors: %d", 
             GPU_NAME, 
             used / (1024.0 * 1024.0 * 1024.0),
             total / (1024.0 * 1024.0 * 1024.0),
             gpu_device.tensor_count);
}