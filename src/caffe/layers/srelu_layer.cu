#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/srelu_layer.hpp"

namespace caffe {

// CUDA kernele for forward
template <typename Dtype>
__global__ void SReLUForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out,
    const Dtype* tr_data, const Dtype* ar_data, const Dtype* tl_data, const Dtype* al_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    if (in[index] >= tr_data[c])
    	out[index] = tr_data[c] + ar_data[c] * (in[index] - tr_data[c]);
    else if ( in[index] > tl_data[c] && in[index] < tr_data[c] )
    	out[index] = in[index];
    else
    	out[index] = tl_data[c] + al_data[c] * (in[index] - tl_data[c]);    
  }
}


// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void SReLUBackward(const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
    const Dtype* tr_data, const Dtype* ar_data, const Dtype* tl_data, const Dtype* al_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = in_diff[index] * 
    	(ar_data[c] * (in_data[index] >= tr_data[c]) + 
    	 (in_data[index] > tl_data[c] && in_data[index] < tr_data[c]) +
         al_data[c] * (in_data[index] <= tl_data[c]));
  }
}


// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void SReLUParamsBackward(const int n,
    const int rows, const int dim, const int rowPitch,
    const Dtype* in_diff, const Dtype* in_data, 
    const Dtype* tr_data, const Dtype* ar_data, const Dtype* tl_data, const Dtype* al_data,
    Dtype* out_diff,
    int param_num) {
  switch (param_num) {
    case 0:
      CUDA_KERNEL_LOOP(index, n) {
        out_diff[index] = in_diff[index] * (1 - ar_data[index/dim]) * (in_data[index] >= tr_data[index/dim]);
        for ( int k = 1; k < rows; k++ ) {
            out_diff[index] += in_diff[index + k*rowPitch]
               * (1 - ar_data[index/dim]) * (in_data[index + k*rowPitch] >= tr_data[index/dim]);
        }
      }
      break;
    case 1:
      CUDA_KERNEL_LOOP(index, n) {
        out_diff[index] = in_diff[index] * (in_data[index] - tr_data[index/dim]) * (in_data[index] >= tr_data[index/dim]);
        for ( int k = 1; k < rows; k++ ) {
            out_diff[index] += in_diff[index + k*rowPitch]
               * (in_data[index + k*rowPitch] - tr_data[index/dim]) * (in_data[index + k*rowPitch] >= tr_data[index/dim]);
        }
      }
      break;
    case 2:
      CUDA_KERNEL_LOOP(index, n) {
        out_diff[index] = in_diff[index] * (1 - al_data[index/dim]) * (in_data[index] <= tl_data[index/dim]);
        for ( int k = 1; k < rows; k++ ) {
            out_diff[index] += in_diff[index + k*rowPitch]
               * (1 - al_data[index/dim]) * (in_data[index + k*rowPitch] <= tl_data[index/dim]);
        }
      }
      break;
    case 3:            
      CUDA_KERNEL_LOOP(index, n) {        
        out_diff[index] = in_diff[index] * (in_data[index] - tl_data[index/dim]) * (in_data[index] <= tl_data[index/dim]);
        for ( int k = 1; k < rows; k++ ) {
            out_diff[index] += in_diff[index + k*rowPitch]
               * (in_data[index + k*rowPitch] - tl_data[index/dim]) * (in_data[index + k*rowPitch] <= tl_data[index/dim]);                
        }
      }
      break;
    default:
      assert(false);
  }
}

template <typename Dtype>
void SReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* tr_data = this->blobs_[0]->gpu_data();
  const Dtype* ar_data = this->blobs_[1]->gpu_data();
  const Dtype* tl_data = this->blobs_[2]->gpu_data();
  const Dtype* al_data = this->blobs_[3]->gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;

  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  SReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data,
      tr_data, ar_data, tl_data, al_data, div_factor);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void SReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* tr_data = this->blobs_[0]->gpu_data();
  const Dtype* ar_data = this->blobs_[1]->gpu_data();
  const Dtype* tl_data = this->blobs_[2]->gpu_data();
  const Dtype* al_data = this->blobs_[3]->gpu_data();
  const int count = bottom[0]->count();
  const int cdim = bottom[0]->count(1);
  const int dim = bottom[0]->count(2);      
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.    
  for (int param_num = 0; param_num < 4; ++param_num) {    
    if (this->param_propagate_down_[param_num]) {
      Dtype* param_diff = this->blobs_[param_num]->mutable_gpu_diff();          
      // compute element-wise diff
      // NOLINT_NEXT_LINE(whitespace/operators)
      SReLUParamsBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
        CAFFE_CUDA_NUM_THREADS>>>(
        cdim, bottom[0]->num(), dim, top[0]->offset(1),
        top_diff, bottom_data,
        tr_data, ar_data, tl_data, al_data,
        backward_buff_.mutable_gpu_diff(),
        param_num);      
      CUDA_POST_KERNEL_CHECK;
      if (channel_shared_) {
        Dtype dsum;
        caffe_gpu_dot<Dtype>(channels * dim, backward_buff_.gpu_diff(),
         multiplier_.gpu_data(), &dsum);
        caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum), param_diff);
      } else {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
          backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
          param_diff);
      }
    }
  }
  
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* slope_data = this->blobs_[0]->gpu_data();
    int div_factor = channel_shared_ ? channels : 1;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, top_diff, bottom_data, bottom_diff,
        tr_data, ar_data, tl_data, al_data,
        div_factor);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(SReLULayer);

}  // namespace caffe
