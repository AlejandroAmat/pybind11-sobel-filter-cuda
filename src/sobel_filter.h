#pragma once
#include "torch/extension.h"

void apply_filter(const torch::Tensor& image, torch::Tensor& output);