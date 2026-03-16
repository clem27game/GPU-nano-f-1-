import ctypes
import numpy as np

# Load C library
lib = ctypes.CDLL('./libgpu_nano_f1.so')

class GPUNanoF1:
    def __init__(self):
        lib.gpu_init()
    
    def create_tensor(self, rows, cols):
        return lib.gpu_tensor_create(rows, cols)
    
    def matmul(self, tensor_a, tensor_b, tensor_c):
        return lib.gpu_matmul(tensor_a, tensor_b, tensor_c)
    
    def relu_forward(self, tensor_in, tensor_out):
        return lib.gpu_relu_forward(tensor_in, tensor_out)
    
    def set_data(self, tensor_id, data):
        flat_data = data.flatten().astype(np.float32)
        lib.gpu_tensor_set_data(tensor_id, flat_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(flat_data))
    
    def get_data(self, tensor_id, shape):
        output = np.zeros(np.prod(shape), dtype=np.float32)
        lib.gpu_tensor_get_data(tensor_id, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(output))
        return output.reshape(shape)
    
    def memory_info(self):
        used = ctypes.c_size_t()
        total = ctypes.c_size_t()
        lib.gpu_memory_info(ctypes.byref(used), ctypes.byref(total))
        return used.value / (1024**3), total.value / (1024**3)
    
    def shutdown(self):
        lib.gpu_shutdown()

# Compilation: gcc -shared -fPIC -fopenmp gpu_nano_f1.c -o libgpu_nano_f1.so
