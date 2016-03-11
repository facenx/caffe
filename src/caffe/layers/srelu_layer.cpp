#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/srelu_layer.hpp"

namespace caffe {

template <typename Dtype>
void SReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  SReLUParameter srelu_param = this->layer_param().srelu_param();
  int channels = bottom[0]->channels();
  channel_shared_ = srelu_param.channel_shared();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(4);
    for ( int i = 0; i < 4; i++ ) {
      if (channel_shared_) {
        this->blobs_[i].reset(new Blob<Dtype>(vector<int>(0)));
      } else {
        this->blobs_[i].reset(new Blob<Dtype>(vector<int>(1, channels)));
      }
    }
    // Initialize learnable parameters
    shared_ptr<Filler<Dtype> > filler;
    // set threshold-right (tr) parameter
    if (srelu_param.has_tr_filler()) {
      filler.reset(GetFiller<Dtype>(srelu_param.tr_filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(1.);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[0].get());
    // set slope-right (ar) parameter
    if (srelu_param.has_ar_filler()) {
      filler.reset(GetFiller<Dtype>(srelu_param.ar_filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(1.);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[1].get());
    // set threshold-left (tl) parameter
    if (srelu_param.has_tl_filler()) {
      filler.reset(GetFiller<Dtype>(srelu_param.tl_filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(0.);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[2].get());
    // set slope-left (al) parameter
    if (srelu_param.has_al_filler()) {
      filler.reset(GetFiller<Dtype>(srelu_param.al_filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(0.25);     // as in DeepID training practice
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[3].get());
  }
  CHECK_GE(this->blobs_[0]->cpu_data()[0], this->blobs_[2]->cpu_data()[0]) << "tr (threshold right) must be >= tl (threshold left) ";

  for ( int i = 0; i < 4; i++ ) {
    if (channel_shared_) {
      CHECK_EQ(this->blobs_[i]->count(), 1)
          << "Negative slope size is inconsistent with prototxt config";
    } else {
      CHECK_EQ(this->blobs_[i]->count(), channels)
          << "Negative slope size is inconsistent with prototxt config";
    }
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
//  multiplier_.Reshape(vector<int>(1, bottom[0]->count(1)));
//  backward_buff_.Reshape(vector<int>(1, bottom[0]->count(1)));
//  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void SReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* tr_data = this->blobs_[0]->cpu_data();
  const Dtype* ar_data = this->blobs_[1]->cpu_data();
  const Dtype* tl_data = this->blobs_[2]->cpu_data();
  const Dtype* al_data = this->blobs_[3]->cpu_data();

  // For in-place computation
  if (bottom[0] == top[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;
  for (int i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    if ( bottom_data[i] >= tr_data[c] )
      top_data[i] = tr_data[c] + ar_data[c]*(bottom_data[i]-tr_data[c]);
    else if ( bottom_data[i] < tr_data[c] && bottom_data[i] > tl_data[c] )
      top_data[i] = bottom_data[i];
    else
      top_data[i] = tl_data[c] + al_data[c]*(bottom_data[i]-tl_data[c]);
  }
}

template <typename Dtype>
void SReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* tr_data = this->blobs_[0]->cpu_data();
  const Dtype* ar_data = this->blobs_[1]->cpu_data();
  const Dtype* tl_data = this->blobs_[2]->cpu_data();
  const Dtype* al_data = this->blobs_[3]->cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.cpu_data();
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;

  // Propagte to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  // . threshold-right (tr) parameter
  if (this->param_propagate_down_[0]) {
    Dtype* tr_diff = this->blobs_[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      tr_diff[c] += top_diff[i] * (1 - ar_data[c]) * (bottom_data[i] >= tr_data[c]);
    }
  }
  // . slope-right (ar) parameter
  if (this->param_propagate_down_[1]) {
    Dtype* ar_diff = this->blobs_[1]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      ar_diff[c] += top_diff[i] * (bottom_data[i] - tr_data[c]) * (bottom_data[i] >= tr_data[c]);
    }
  }
  // . threshold-left (tl) parameter
  if (this->param_propagate_down_[2]) {
    Dtype* tl_diff = this->blobs_[2]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      tl_diff[c] += top_diff[i] * (1 - al_data[c]) * (bottom_data[i] <= tl_data[c]);
    }
  }
  // . slope-left (al) parameter
  if (this->param_propagate_down_[3]) {
    Dtype* al_diff = this->blobs_[3]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      al_diff[c] += top_diff[i] * (bottom_data[i] - tl_data[c]) * (bottom_data[i] <= tl_data[c]);
    }
  }

  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      bottom_diff[i] = top_diff[i] *
          ( ar_data[c] * (bottom_data[i] >= tr_data[c]) +
            Dtype(bottom_data[i] < tr_data[c] && bottom_data[i] > tl_data[c]) +
            al_data[c] * (bottom_data[i] <= tl_data[c]) );
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(PReLULayer);
#endif

INSTANTIATE_CLASS(SReLULayer);
REGISTER_LAYER_CLASS(SReLU);

}  // namespace caffe
