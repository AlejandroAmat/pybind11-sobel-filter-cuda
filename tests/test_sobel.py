import torch
import cv2
import matplotlib.pyplot as plt
import time
import sys
import os

# Add parent directory to path to import the compiled module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sobel_filter_cuda


def test_grayscale():
    # Load grayscale image
    img = cv2.imread("test_images/lena.png", cv2.IMREAD_GRAYSCALE)

    # Convert to PyTorch tensor (H,W) -> (1,H,W)
    img_tensor = (
        torch.from_numpy(img.astype("float32") / 255.0).unsqueeze(0).contiguous()
    )
    img_gpu = img_tensor.cuda()

    # Save original for display
    original_gpu = img_gpu.clone()

    # Warmup phase
    for _ in range(5):
        sobel_filter_cuda.sobel_filter(img_gpu.clone())

    # Time the Sobel filter (returns new tensor)
    torch.cuda.synchronize()
    start_time = time.time()

    output = sobel_filter_cuda.sobel_filter(img_gpu)

    torch.cuda.synchronize()
    end_time = time.time()

    execution_time = (end_time - start_time) * 1000  # ms
    execution_time_sec = end_time - start_time  # seconds

    # Test in-place version
    output_inplace = torch.zeros_like(img_gpu)

    # Warmup for in-place
    for _ in range(5):
        temp_output = torch.zeros_like(img_gpu)
        sobel_filter_cuda.sobel_filter_(img_gpu, temp_output)

    torch.cuda.synchronize()
    start_time_inplace = time.time()

    sobel_filter_cuda.sobel_filter_(img_gpu, output_inplace)

    torch.cuda.synchronize()
    end_time_inplace = time.time()

    execution_time_inplace = (end_time_inplace - start_time_inplace) * 1000  # ms
    execution_time_inplace_sec = end_time_inplace - start_time_inplace  # seconds

    # Show results with more white space
    plt.figure(figsize=(18, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(original_gpu[0].cpu().numpy(), cmap="gray")
    plt.title("Original Grayscale\n(256x256)", fontsize=12, pad=20)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(output[0].cpu().numpy(), cmap="gray")
    plt.title(
        f"Sobel Edges (New Tensor)\nTime: {execution_time:.3f} ms\n({execution_time_sec:.6f} s)",
        fontsize=12,
        pad=20,
    )
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(output_inplace[0].cpu().numpy(), cmap="gray")
    plt.title(
        f"Sobel Edges (In-place)\nTime: {execution_time_inplace:.3f} ms\n({execution_time_inplace_sec:.6f} s)",
        fontsize=12,
        pad=20,
    )
    plt.axis("off")

    plt.tight_layout(pad=3.0)
    plt.savefig("grayscale_sobel_result.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(
        f"Grayscale execution time (new tensor): {execution_time:.3f} ms ({execution_time_sec:.6f} s)"
    )
    print(
        f"Grayscale execution time (in-place): {execution_time_inplace:.3f} ms ({execution_time_inplace_sec:.6f} s)"
    )


def test_rgb():
    # Load RGB image
    img_bgr = cv2.imread("test_images/img1.jpg")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to PyTorch tensor (H,W,C) -> (C,H,W)
    img_tensor = (
        torch.from_numpy(img_rgb.astype("float32") / 255.0)
        .permute(2, 0, 1)
        .contiguous()
    )
    img_gpu = img_tensor.cuda()

    # Warmup phase
    for _ in range(5):
        sobel_filter_cuda.sobel_filter(img_gpu.clone())

    # Time the Sobel filter (returns new tensor)
    torch.cuda.synchronize()
    start_time = time.time()

    output = sobel_filter_cuda.sobel_filter(img_gpu)

    torch.cuda.synchronize()
    end_time = time.time()

    execution_time = (end_time - start_time) * 1000  # ms
    execution_time_sec = end_time - start_time  # seconds

    # Test in-place version
    output_inplace = torch.zeros_like(img_gpu)

    # Warmup for in-place
    for _ in range(5):
        temp_output = torch.zeros_like(img_gpu)
        sobel_filter_cuda.sobel_filter_(img_gpu, temp_output)

    torch.cuda.synchronize()
    start_time_inplace = time.time()

    sobel_filter_cuda.sobel_filter_(img_gpu, output_inplace)

    torch.cuda.synchronize()
    end_time_inplace = time.time()

    execution_time_inplace = (end_time_inplace - start_time_inplace) * 1000  # ms
    execution_time_inplace_sec = end_time_inplace - start_time_inplace  # seconds

    # Show results with more white space
    plt.figure(figsize=(18, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title(
        f"Original RGB\n({img_rgb.shape[1]}x{img_rgb.shape[0]})", fontsize=12, pad=20
    )
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(output.permute(1, 2, 0).cpu().numpy())
    plt.title(
        f"Sobel Edges (New Tensor)\nTime: {execution_time:.3f} ms\n({execution_time_sec:.6f} s)",
        fontsize=12,
        pad=20,
    )
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(output_inplace.permute(1, 2, 0).cpu().numpy())
    plt.title(
        f"Sobel Edges (In-place)\nTime: {execution_time_inplace:.3f} ms\n({execution_time_inplace_sec:.6f} s)",
        fontsize=12,
        pad=20,
    )
    plt.axis("off")

    plt.tight_layout(pad=3.0)
    plt.savefig("rgb_sobel_result.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(
        f"RGB execution time (new tensor): {execution_time:.3f} ms ({execution_time_sec:.6f} s)"
    )
    print(
        f"RGB execution time (in-place): {execution_time_inplace:.3f} ms ({execution_time_inplace_sec:.6f} s)"
    )


if __name__ == "__main__":
    print("Testing Sobel Filter Performance")
    print("================================")

    print("\n1. Grayscale test:")
    test_grayscale()

    print("\n2. RGB test:")
    test_rgb()
