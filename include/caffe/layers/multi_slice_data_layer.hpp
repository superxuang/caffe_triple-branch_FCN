#ifndef CAFFE_MULTI_SLICE_DATA_LAYER_HPP_
#define CAFFE_MULTI_SLICE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <itkImage.h>
#include <itkSmartPointer.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MultiSliceDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  typedef float VoxelType;
  typedef itk::Image<VoxelType, 3> ImageType;
  typedef itk::Image<char, 3> LabelType;
  typedef itk::Image<VoxelType, 2> ImageSliceType;
  typedef itk::Image<char, 2> LabelSliceType;

  struct SliceRecord {
	  int file_num_;
	  itk::SmartPointer<ImageSliceType> data_;
	  std::vector<int> label_;
  };

  explicit MultiSliceDataLayer(const LayerParameter& param)
	  : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~MultiSliceDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiSliceData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);
  vector<SliceRecord*> slice_list_;
  vector<int> slice_id_list_;
  int contour_num_;
  int slice_id_;
};

}  // namespace caffe

#endif  // CAFFE_MULTI_SLICE_DATA_LAYER_HPP_
