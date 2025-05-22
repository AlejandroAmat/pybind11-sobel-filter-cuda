#include <torch/extension.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include "sobel_filter.h"


torch::Tensor sobel_filter_cuda(const torch::Tensor& image) {
    TORCH_CHECK(image.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(image.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(image.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(image.dim() == 3, "Input tensor must be 3D (C, H, W)");
    
    int64_t channels = image.size(0);
    TORCH_CHECK(channels == 1 || channels == 3, 
                "Only 1 or 3 channels supported, got: " + std::to_string(channels));
    
    TORCH_CHECK(image.size(1) >= 3 && image.size(2) >= 3, 
                "Image must be at least 3x3 pixels for Sobel filter");
    
    // Create separate output tensor
    torch::Tensor output = torch::zeros_like(image);
    
    apply_filter(image, output);
    
    return output;
}

void sobel_filter_cuda_inplace(torch::Tensor& image, torch::Tensor& output) {
    TORCH_CHECK(image.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(output.is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(image.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "Output tensor must be contiguous");
    TORCH_CHECK(image.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Output tensor must be float32");
    TORCH_CHECK(image.dim() == 3, "Input tensor must be 3D (C, H, W)");
    TORCH_CHECK(output.dim() == 3, "Output tensor must be 3D (C, H, W)");
    
    // Check shapes match
    TORCH_CHECK(image.sizes() == output.sizes(), "Input and output tensors must have same shape");
    
    int64_t channels = image.size(0);
    TORCH_CHECK(channels == 1 || channels == 3, 
                "Only 1 or 3 channels supported, got: " + std::to_string(channels));
    
    TORCH_CHECK(image.size(1) >= 3 && image.size(2) >= 3, 
                "Image must be at least 3x3 pixels for Sobel filter");
    
    apply_filter(image, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sobel_filter", &sobel_filter_cuda, 
          "Apply Sobel edge detection filter (CUDA implementation)",
          py::arg("image"));
    
    m.def("sobel_filter_", &sobel_filter_cuda_inplace, 
          "Apply Sobel edge detection filter with separate output (CUDA implementation)",
          py::arg("image"), py::arg("output"));
}