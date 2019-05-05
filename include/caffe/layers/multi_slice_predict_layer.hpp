#ifndef CAFFE_MULTI_SLICE_PREDICT_LAYER_HPP_
#define CAFFE_MULTI_SLICE_PREDICT_LAYER_HPP_
#define OUTPUT_PROJECTION

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <itkImage.h>
#include <itkSmartPointer.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

namespace caffe {

template <typename Dtype>
class MultiSlicePredictLayer : public Layer<Dtype> {
 public:
  typedef itk::Image<Dtype, 3> ImageType;
  typedef itk::Image<char, 3> LabelType;
  typedef itk::Image<Dtype, 2> ImageSliceType;
  typedef itk::Image<char, 2> LabelSliceType;

  explicit MultiSlicePredictLayer(const LayerParameter& param)
	  : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiSlicePredict"; }

  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 protected:
  vector<std::string> file_names_;
  int line_id_;
};


}  // namespace caffe

#endif  // CAFFE_MULTI_SLICE_PREDICT_LAYER_HPP_
