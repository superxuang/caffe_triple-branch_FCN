#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling3d_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void Pooling3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Pooling3DParameter pool_param = this->layer_param_.pooling_3d_param();
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w() || pool_param.has_kernel_l()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w() && pool_param.has_kernel_l()))
      << "Filter size is kernel_size OR kernel_h and kernel_w and kernel_l; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w() && pool_param.has_kernel_l()))
      << "For non-square filters both kernel_h and kernel_w and kernel_l are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
	  && pool_param.has_pad_w() && pool_param.has_pad_l())
	  || (!pool_param.has_pad_h() && !pool_param.has_pad_w() && !pool_param.has_pad_l()))
      << "pad is pad OR pad_h and pad_w and pad_l are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
	  && pool_param.has_stride_w() && pool_param.has_stride_l())
	  || (!pool_param.has_stride_h() && !pool_param.has_stride_w() && !pool_param.has_stride_l()))
      << "Stride is stride OR stride_h and stride_w and stride_l are required.";
  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
    kernel_l_ = bottom[0]->shape(2);
    kernel_h_ = bottom[0]->shape(3);
    kernel_w_ = bottom[0]->shape(4);
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_l_ = kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_l_ = pool_param.kernel_l();
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_l_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_l()) {
    pad_l_ = pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_l_ = pool_param.pad_l();
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_l()) {
    stride_l_ = stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_l_ = pool_param.stride_l();
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_l_ == 0 && pad_h_ == 0 && pad_w_ == 0 && stride_l_ == 1 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_l_ != 0 || pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
	CHECK_LT(pad_l_, kernel_l_);
	CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
}

template <typename Dtype>
void Pooling3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(5, bottom[0]->num_axes()) << "Input must have 5 axes, "
      << "corresponding to (num, channels, length, height, width)";
  channels_ = bottom[0]->channels();
  length_ = bottom[0]->shape(2);
  height_ = bottom[0]->shape(3);
  width_ = bottom[0]->shape(4);
  if (global_pooling_) {
    kernel_l_ = bottom[0]->shape(2);
	kernel_h_ = bottom[0]->shape(3);
	kernel_w_ = bottom[0]->shape(4);
  }
  pooled_length_ = static_cast<int>(ceil(static_cast<float>(
	  length_ + 2 * pad_l_ - kernel_l_) / stride_l_)) + 1;
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_l_ || pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_length_ - 1) * stride_l_ >= length_ + pad_l_) {
      --pooled_length_;
    }
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
	CHECK_LT((pooled_length_ - 1) * stride_l_, length_ + pad_l_);
	CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  std::vector<int> shapes(5);
  shapes[0] = bottom[0]->num();
  shapes[1] = channels_;
  shapes[2] = pooled_length_;
  shapes[3] = pooled_height_;
  shapes[4] = pooled_width_;
  top[0]->Reshape(shapes);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(shapes);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(shapes);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void Pooling3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int pl = 0; pl < pooled_length_; ++pl) {
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              int lstart = pl * stride_l_ - pad_l_;
              int hstart = ph * stride_h_ - pad_h_;
              int wstart = pw * stride_w_ - pad_w_;
			  int lend = min(lstart + kernel_l_, length_);
			  int hend = min(hstart + kernel_h_, height_);
              int wend = min(wstart + kernel_w_, width_);
			  lstart = max(lstart, 0);
			  hstart = max(hstart, 0);
              wstart = max(wstart, 0);
              const int pool_index = pl * pooled_height_ * pooled_width_ + ph * pooled_width_ + pw;
              for (int l = lstart; l < lend; ++l) {
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    const int index = l * height_ * width_ + h * width_ + w;
                    if (bottom_data[index] > top_data[pool_index]) {
                      top_data[pool_index] = bottom_data[index];
                      if (use_top_mask) {
                        top_mask[pool_index] = static_cast<Dtype>(index);
                      } else {
                        mask[pool_index] = index;
                      }
                    }
                  }
                }
			  }
            }
		  }
        }
        // compute offset
		std::vector<int> offsets(5);
		offsets[0] = 0;
		offsets[1] = 1;
		offsets[2] = 0;
		offsets[3] = 0;
		offsets[4] = 0;
		bottom_data += bottom[0]->offset(offsets);
		top_data += top[0]->offset(offsets);
        if (use_top_mask) {
          top_mask += top[0]->offset(offsets);
        } else {
          mask += top[0]->offset(offsets);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int pl = 0; pl < pooled_length_; ++pl) {
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              int lstart = pl * stride_l_ - pad_l_;
              int hstart = ph * stride_h_ - pad_h_;
              int wstart = pw * stride_w_ - pad_w_;
			  int lend = min(lstart + kernel_l_, length_ + pad_l_);
			  int hend = min(hstart + kernel_h_, height_ + pad_h_);
              int wend = min(wstart + kernel_w_, width_ + pad_w_);
              int pool_size = (lend - lstart) * (hend - hstart) * (wend - wstart);
			  lstart = max(lstart, 0);
			  hstart = max(hstart, 0);
              wstart = max(wstart, 0);
			  lend = min(lend, length_);
			  hend = min(hend, height_);
              wend = min(wend, width_);
              for (int l = lstart; l < lend; ++l) {
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    top_data[pl * pooled_height_ * pooled_width_ + ph * pooled_width_ + pw] +=
                        bottom_data[l * height_ * width_ + h * width_ + w];
                  }
                }
			  }
			  top_data[pl * pooled_height_ * pooled_width_ + ph * pooled_width_ + pw] /= pool_size;
            }
          }
		}
        // compute offset
		std::vector<int> offsets(5);
		offsets[0] = 0;
		offsets[1] = 1;
		offsets[2] = 0;
		offsets[3] = 0;
		offsets[4] = 0;
		bottom_data += bottom[0]->offset(offsets);
		top_data += top[0]->offset(offsets);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void Pooling3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int pl = 0; pl < pooled_length_; ++pl) {
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              const int index = pl * pooled_height_ * pooled_width_ + ph * pooled_width_ + pw;
              const int bottom_index =
                  use_top_mask ? top_mask[index] : mask[index];
              bottom_diff[bottom_index] += top_diff[index];
            }
          }
		}
		std::vector<int> offsets(5);
		offsets[0] = 0;
		offsets[1] = 1;
		offsets[2] = 0;
		offsets[3] = 0;
		offsets[4] = 0;
		bottom_diff += bottom[0]->offset(offsets);
		top_diff += top[0]->offset(offsets);
        if (use_top_mask) {
          top_mask += top[0]->offset(offsets);
        } else {
          mask += top[0]->offset(offsets);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int pl = 0; pl < pooled_length_; ++pl) {
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              int lstart = pl * stride_l_ - pad_l_;
              int hstart = ph * stride_h_ - pad_h_;
              int wstart = pw * stride_w_ - pad_w_;
			  int lend = min(lstart + kernel_l_, length_ + pad_l_);
			  int hend = min(hstart + kernel_h_, height_ + pad_h_);
              int wend = min(wstart + kernel_w_, width_ + pad_w_);
              int pool_size = (lend - lstart) * (hend - hstart) * (wend - wstart);
			  lstart = max(lstart, 0);
			  hstart = max(hstart, 0);
              wstart = max(wstart, 0);
			  lend = min(lend, length_);
			  hend = min(hend, height_);
              wend = min(wend, width_);
              for (int l = lstart; l < lend; ++l) {
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    bottom_diff[l * height_ * width_ + h * width_ + w] +=
                      top_diff[pl * pooled_height_ * pooled_width_ + ph * pooled_width_ + pw] / pool_size;
                  }
                }
			  }
            }
          }
		}
        // offset
		std::vector<int> offsets(5);
		offsets[0] = 0;
		offsets[1] = 1;
		offsets[2] = 0;
		offsets[3] = 0;
		offsets[4] = 0;
		bottom_diff += bottom[0]->offset(offsets);
		top_diff += top[0]->offset(offsets);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(Pooling3DLayer);
#endif

INSTANTIATE_CLASS(Pooling3DLayer);
REGISTER_LAYER_CLASS(Pooling3D);

}  // namespace caffe
