import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from scipy import ndimage

import sobel_filter_cuda


def numpy_sobel(image):
    """Pure NumPy Sobel implementation (slowest)"""
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # Manual convolution
    height, width = image.shape
    grad_x = np.zeros_like(image)
    grad_y = np.zeros_like(image)

    # Apply convolution manually (very slow)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Extract 3x3 patch
            patch = image[y - 1 : y + 2, x - 1 : x + 2]

            # Apply Sobel kernels
            grad_x[y, x] = np.sum(patch * sobel_x)
            grad_y[y, x] = np.sum(patch * sobel_y)

    # Calculate magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude


def opencv_sobel(image):
    """OpenCV Sobel implementation (CPU optimized)"""
    # Convert to uint8 for OpenCV
    img_uint8 = (image * 255).astype(np.uint8)

    # Apply Sobel filters
    sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize to [0, 1]
    magnitude = magnitude / 255.0
    return magnitude


def cuda_sobel(image_gpu, output):
    """CUDA Sobel implementation (fastest) - in-place version"""
    sobel_filter_cuda.sobel_filter_(image_gpu, output)
    return output


def benchmark_lena():
    """Compare NumPy, OpenCV, and CUDA on Lena image"""
    print("NumPy vs OpenCV vs CUDA Sobel Comparison")
    print("Testing on Lena (256x256 grayscale)")
    print("=" * 50)

    # Load Lena image
    img_cv = cv2.imread("test_images/lena.png", cv2.IMREAD_GRAYSCALE)
    img_float = img_cv.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_float).unsqueeze(0).contiguous().cuda()

    height, width = img_float.shape
    total_pixels = height * width
    print(f"Image size: {width}x{height} ({total_pixels:,} pixels)")

    results = {}

    # 1. Benchmark NumPy (Pure Python - slowest)
    print("\nTesting NumPy (pure Python)...")

    # Warmup
    _ = numpy_sobel(img_float)

    # Time NumPy
    start_time = time.time()
    numpy_result = numpy_sobel(img_float)
    end_time = time.time()

    numpy_time = (end_time - start_time) * 1000  # ms
    numpy_throughput = total_pixels / (end_time - start_time) / 1e6  # MP/s

    results["numpy"] = {
        "time": numpy_time,
        "throughput": numpy_throughput,
        "result": numpy_result,
    }

    print(f"NumPy:    {numpy_time:.1f} ms ({numpy_throughput:.3f} MP/s)")

    # 2. Benchmark OpenCV (CPU optimized)
    print("\nTesting OpenCV (CPU optimized)...")

    # Warmup
    for _ in range(3):
        _ = opencv_sobel(img_float)

    # Time OpenCV
    start_time = time.time()
    num_runs = 10
    for _ in range(num_runs):
        opencv_result = opencv_sobel(img_float)
    end_time = time.time()

    opencv_time = (end_time - start_time) / num_runs * 1000  # ms
    opencv_throughput = (
        total_pixels / ((end_time - start_time) / num_runs) / 1e6
    )  # MP/s

    results["opencv"] = {
        "time": opencv_time,
        "throughput": opencv_throughput,
        "result": opencv_result,
    }

    print(f"OpenCV:   {opencv_time:.3f} ms ({opencv_throughput:.1f} MP/s)")

    # 3. Benchmark CUDA (GPU optimized)
    print("\nTesting CUDA (GPU optimized)...")
    output = torch.zeros_like(img_tensor)
    # Warmup
    for _ in range(5):
        _ = cuda_sobel(img_tensor, output)

    # Time CUDA
    torch.cuda.synchronize()
    start_time = time.time()

    num_runs = 100  # Many runs for accurate GPU timing
    for _ in range(num_runs):
        cuda_result = cuda_sobel(img_tensor, output)

    torch.cuda.synchronize()
    end_time = time.time()

    cuda_time = (end_time - start_time) / num_runs * 1000  # ms
    cuda_throughput = total_pixels / ((end_time - start_time) / num_runs) / 1e6  # MP/s

    results["cuda"] = {
        "time": cuda_time,
        "throughput": cuda_throughput,
        "result": cuda_result[0].cpu().numpy(),
    }

    print(f"CUDA:     {cuda_time:.3f} ms ({cuda_throughput:.1f} MP/s)")

    # Calculate speedups
    numpy_vs_opencv = numpy_time / opencv_time
    numpy_vs_cuda = numpy_time / cuda_time
    opencv_vs_cuda = opencv_time / cuda_time

    # Print comparison
    print("\n" + "=" * 50)
    print("SPEEDUP COMPARISON")
    print("=" * 50)
    print(f"NumPy:    {numpy_time:8.1f} ms (baseline)")
    print(
        f"OpenCV:   {opencv_time:8.3f} ms ({numpy_vs_opencv:6.1f}x faster than NumPy)"
    )
    print(f"CUDA:     {cuda_time:8.3f} ms ({numpy_vs_cuda:6.0f}x faster than NumPy)")
    print(f"                            ({opencv_vs_cuda:6.1f}x faster than OpenCV)")

    # Create visualization
    create_lena_comparison(results, numpy_vs_opencv, numpy_vs_cuda, opencv_vs_cuda)

    return results


def create_lena_comparison(results, numpy_vs_opencv, numpy_vs_cuda, opencv_vs_cuda):
    """Create comprehensive comparison visualization"""

    # Load original image for display
    img_cv = cv2.imread("test_images/lena.png", cv2.IMREAD_GRAYSCALE)
    img_original = img_cv.astype(np.float32) / 255.0

    # 1. Results comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original image
    axes[0, 0].imshow(img_original, cmap="gray")
    axes[0, 0].set_title("Original Lena (256x256)", fontsize=12)
    axes[0, 0].axis("off")

    # NumPy result
    axes[0, 1].imshow(results["numpy"]["result"], cmap="gray")
    axes[0, 1].set_title(f'NumPy Sobel\n{results["numpy"]["time"]:.1f} ms', fontsize=12)
    axes[0, 1].axis("off")

    # OpenCV result
    axes[1, 0].imshow(results["opencv"]["result"], cmap="gray")
    axes[1, 0].set_title(
        f'OpenCV Sobel\n{results["opencv"]["time"]:.3f} ms\n({numpy_vs_opencv:.1f}x faster)',
        fontsize=12,
    )
    axes[1, 0].axis("off")

    # CUDA result
    axes[1, 1].imshow(results["cuda"]["result"], cmap="gray")
    axes[1, 1].set_title(
        f'CUDA Sobel (In-place)\n{results["cuda"]["time"]:.3f} ms\n({numpy_vs_cuda:.0f}x faster)',
        fontsize=12,
    )
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("lena_three_way_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 2. Performance bar chart
    plt.figure(figsize=(12, 8))

    methods = [
        "NumPy\n(Pure Python)",
        "OpenCV\n(CPU Optimized)",
        "CUDA\n(GPU In-place)",
    ]
    times = [
        results["numpy"]["time"],
        results["opencv"]["time"],
        results["cuda"]["time"],
    ]
    colors = ["red", "orange", "green"]

    bars = plt.bar(methods, times, color=colors, alpha=0.7)

    plt.xlabel("Implementation")
    plt.ylabel("Execution Time (ms)")
    plt.title("Sobel Edge Detection Performance Comparison\nLena 256x256 Grayscale")
    plt.yscale("log")

    # Add value labels on bars
    for i, (bar, time_val) in enumerate(zip(bars, times)):
        if i == 0:  # NumPy
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{time_val:.1f} ms",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        else:  # OpenCV and CUDA
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{time_val:.3f} ms",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # Add speedup annotations
    plt.annotate(
        f"{numpy_vs_opencv:.1f}x faster",
        xy=(1, results["opencv"]["time"]),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="blue"),
    )

    plt.annotate(
        f"{numpy_vs_cuda:.0f}x faster",
        xy=(2, results["cuda"]["time"]),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="blue"),
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("performance_three_way_chart.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 3. Throughput comparison
    plt.figure(figsize=(10, 6))

    throughputs = [
        results["numpy"]["throughput"],
        results["opencv"]["throughput"],
        results["cuda"]["throughput"],
    ]

    bars = plt.bar(methods, throughputs, color=colors, alpha=0.7)
    plt.xlabel("Implementation")
    plt.ylabel("Throughput (Megapixels/second)")
    plt.title("Processing Throughput Comparison")
    plt.yscale("log")

    # Add value labels
    for bar, throughput in zip(bars, throughputs):
        if throughput < 1:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{throughput:.3f} MP/s",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        else:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{throughput:.1f} MP/s",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("throughput_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    try:
        print("Starting three-way performance comparison...")
        results = benchmark_lena()
        print("\nComparison complete! Check the generated plots:")
        print("- lena_three_way_comparison.png")
        print("- performance_three_way_chart.png")
        print("- throughput_comparison.png")
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback

        traceback.print_exc()
