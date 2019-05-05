#ifndef CAFFE_MHD_SLICE_DATA_LAYER_HPP_
#define CAFFE_MHD_SLICE_DATA_LAYER_HPP_

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
class MHDSliceDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
	 typedef float VoxelType;
	 typedef itk::Image<VoxelType, 3> ImageType;
	 typedef itk::Image<char, 3> LabelType;
	 typedef itk::Image<VoxelType, 2> ImageSliceType;
	 typedef itk::Image<char, 2> LabelSliceType;

	 struct VolumeRecord {
		 int file_num_;
		 int direct_;
		 itk::SmartPointer<ImageType> data_;
		 std::vector<int> bbox_start_pos_;
		 std::vector<int> bbox_end_pos_;
		 std::vector<bool> organ_exist_;
		 int slice_num_;
		 int slice_size_[2];
		 double slice_spacing_[2];
		 double resample_spacing_;
		 double resample_origin_;
		 double origin_spacing_;
		 double origin_origin_;
	 };

  explicit MHDSliceDataLayer(const LayerParameter& param)
	  : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~MHDSliceDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MHDSliceData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
	 shared_ptr<Caffe::RNG> prefetch_rng_;
	 virtual void ShuffleImages();
	 virtual void load_batch(Batch<Dtype>* batch);
	 vector<VolumeRecord*> volume_list_;
	 vector<int> volume_id_list_;
	 int contour_num_;
	 int volume_id_;
	 int start_slice_id_;
};

}  // namespace caffe

#endif  // CAFFE_MHD_SLICE_DATA_LAYER_HPP_
