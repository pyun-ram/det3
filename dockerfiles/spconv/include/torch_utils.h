// Copyright 2019 Yan Yan
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <tensorview/tensorview.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#ifdef SPCONV_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

namespace tv {
  
#ifdef SPCONV_CUDA
struct TorchGPU: public tv::GPU {
  virtual cudaStream_t getStream() const override {
    return at::cuda::getCurrentCUDAStream();
  }
};
#endif
template <typename T> void check_torch_dtype(const torch::Tensor &tensor) {
  switch (tensor.type().scalarType()) {
  case at::ScalarType::Double: {
    auto val = std::is_same<std::remove_const_t<T>, double>::value;
    TV_ASSERT_RT_ERR(val, "error");
    break;
  }
  case at::ScalarType::Float: {
    auto val = std::is_same<std::remove_const_t<T>, float>::value;
    TV_ASSERT_RT_ERR(val, "error");
    break;
  }
  case at::ScalarType::Int: {
    auto val = std::is_same<std::remove_const_t<T>, int>::value;
    TV_ASSERT_RT_ERR(val, "error");
    break;
  }
  case at::ScalarType::Half: {
    auto val = std::is_same<std::remove_const_t<T>, at::Half>::value;
    TV_ASSERT_RT_ERR(val, "error");
    break;
  }
  case at::ScalarType::Long: {
    auto val = std::is_same<std::remove_const_t<T>, long>::value;
    TV_ASSERT_RT_ERR(val, "error");
    break;
  }
  default:
    TV_ASSERT_RT_ERR(false, "error");
  }
}

template <typename T>
constexpr auto type2torch(T val=T()) -> decltype(torch::kInt32){
  TV_ASSERT_RT_ERR(false, "unknown type");
}

template <>
constexpr auto type2torch(int val) -> decltype(torch::kInt32){
  return torch::kInt32;
}

template <>
constexpr auto type2torch(long val) -> decltype(torch::kInt32){
  return torch::kInt64;
}

template <>
constexpr auto type2torch(float val) -> decltype(torch::kInt32){
  return torch::kFloat32;
}

template <>
constexpr auto type2torch(double val) -> decltype(torch::kInt32){
  return torch::kFloat64;
}

template <typename T>
tv::TensorView<T> torch2tv(const torch::Tensor &tensor) {
  check_torch_dtype<T>(tensor);
  tv::Shape shape;
  for (auto i : tensor.sizes()) {
    shape.push_back(i);
  }
  return tv::TensorView<T>(tensor.data<std::remove_const_t<T>>(), shape);
}
} // namespace tv