#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

// Implement forward pass on GPU
template <typename Dtype>
void WindowPoseDataLayerRegression<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  // First, join the thread
  this->JoinPrefetchThread();
  
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  caffe_copy(this->prefetch_labels_.count(), this->prefetch_labels_.cpu_data(),
      top[1]->mutable_gpu_data());
  // caffe_copy(this->prefetch_sin_azimuth_.count(), this->prefetch_sin_azimuth_.cpu_data(),
  //     top[1]->mutable_gpu_data());
  // caffe_copy(this->prefetch_cos_azimuth_.count(), this->prefetch_cos_azimuth_.cpu_data(),
  //     top[2]->mutable_gpu_data());
  // caffe_copy(this->prefetch_sin_elevation_.count(), this->prefetch_sin_elevation_.cpu_data(),
  //     top[3]->mutable_gpu_data());
  // caffe_copy(this->prefetch_cos_elevation_.count(), this->prefetch_cos_elevation_.cpu_data(),
  //     top[4]->mutable_gpu_data());

  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(WindowPoseDataLayerRegression);

}  // namespace caffe
