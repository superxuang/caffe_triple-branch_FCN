#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "direct.h"

#include "caffe/layers/rpn_output_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <itkResampleImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkIdentityTransform.h>

using std::sqrt;
using std::floor;
using std::ceil;
using std::min;
using std::max;
using std::exp;
using std::log;

namespace caffe {

template <typename Dtype>
void RPNBBoxSegLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	const unsigned int rng_seed = caffe_rng_rand();
	rng_.reset(new Caffe::RNG(rng_seed));

	const RPNOutputParameter& rpn_output_param = this->layer_param_.rpn_output_param();

	const string& source = rpn_output_param.source();
	std::ifstream infile(source.c_str());
	string line;
	size_t pos1, pos2;
	while (std::getline(infile, line)) {
		pos1 = line.find_first_of(' ');
		pos2 = line.find_last_of(' ');
		lines_.push_back(std::make_pair(line.substr(0, pos1),
			std::make_pair(line.substr(pos1 + 1, pos2 - pos1 - 1), line.substr(pos2 + 1))));
	}

	lines_id_ = 0;
}

template <typename Dtype>
void RPNBBoxSegLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	// Reshaping happens during the call to forward.
}

template <typename Dtype>
void RPNBBoxSegLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	const RPNOutputParameter& rpn_output_param = this->layer_param_.rpn_output_param();

	// bottom[0] : bounding box coordinates. n * [batch_id, x1, y1, z1, x2, y2, z2]
	// bottom[1] : bounding box predicted label. n * [label]
	// bottom[2] : predicted segmentation mask in the bounding box.
	// bottom[3] : image information.

	const int bbox_num = bottom[0]->shape(0);
	const Dtype* bbox = bottom[0]->cpu_data();
	const Dtype* bbox_label = bottom[1]->cpu_data();

	int bbox_size[3];
	bbox_size[0] = bottom[2]->shape(4);
	bbox_size[1] = bottom[2]->shape(3);
	bbox_size[2] = bottom[2]->shape(2);
	const Dtype* bbox_data = bottom[2]->cpu_data();

	CHECK_EQ(bbox_num, bottom[2]->shape(0));

	const Dtype* image_info = bottom[3]->cpu_data();
	const int image_size[3] = { (int)image_info[2], (int)image_info[1], (int)image_info[0] };
	const double image_spacing[3] = { image_info[5], image_info[4], image_info[3] };
	const double image_origin[3] = { image_info[8], image_info[7], image_info[6] };
	const int image_origin_size[3] = { (int)image_info[11], (int)image_info[10], (int)image_info[9] };
	const double image_origin_spacing[3] = { image_info[14], image_info[13], image_info[12] };
	const double image_origin_origin[3] = { image_info[17], image_info[16], image_info[15] };
	const int image_scale = image_info[18];

	typedef itk::Image<Dtype, 3> MaskType;
	MaskType::SizeType mask_size;
	mask_size[0] = image_size[0];
	mask_size[1] = image_size[1];
	mask_size[2] = image_size[2];
	MaskType::RegionType mask_region;
	mask_region.SetSize(mask_size);
	MaskType::SpacingType mask_spacing;
	mask_spacing[0] = image_spacing[0];
	mask_spacing[1] = image_spacing[1];
	mask_spacing[2] = image_spacing[2];
	MaskType::PointType mask_origin;
	mask_origin[0] = image_origin[0];
	mask_origin[1] = image_origin[1];
	mask_origin[2] = image_origin[2];

	MaskType::SizeType output_size;
	output_size[0] = image_origin_size[0];
	output_size[1] = image_origin_size[1];
	output_size[2] = image_origin_size[2];
	MaskType::RegionType output_region;
	output_region.SetSize(output_size);
	MaskType::SpacingType output_spacing;
	output_spacing[0] = image_origin_spacing[0];
	output_spacing[1] = image_origin_spacing[1];
	output_spacing[2] = image_origin_spacing[2];
	MaskType::PointType output_origin;
	output_origin[0] = image_origin_origin[0];
	output_origin[1] = image_origin_origin[1];
	output_origin[2] = image_origin_origin[2];

	for (int bbox_id = 0; bbox_id < bbox_num; ++bbox_id)
	{
		MaskType::Pointer mask_roi = MaskType::New();
		MaskType::SizeType mask_roi_size;
		mask_roi_size[0] = bbox_size[0];
		mask_roi_size[1] = bbox_size[1];
		mask_roi_size[2] = bbox_size[2];
		MaskType::RegionType mask_roi_region;
		mask_roi_region.SetSize(mask_roi_size);
		MaskType::SpacingType mask_roi_spacing;
		mask_roi_spacing[0] = (bbox[bbox_id * 7 + 4] - bbox[bbox_id * 7 + 1] + 1) * image_spacing[0] / bbox_size[0];
		mask_roi_spacing[1] = (bbox[bbox_id * 7 + 5] - bbox[bbox_id * 7 + 2] + 1) * image_spacing[1] / bbox_size[1];
		mask_roi_spacing[2] = (bbox[bbox_id * 7 + 6] - bbox[bbox_id * 7 + 3] + 1) * image_spacing[2] / bbox_size[2];
		MaskType::PointType mask_roi_origin;
		mask_roi_origin[0] = bbox[bbox_id * 7 + 1] * image_spacing[0] + image_origin[0];
		mask_roi_origin[1] = bbox[bbox_id * 7 + 2] * image_spacing[1] + image_origin[1];
		mask_roi_origin[2] = bbox[bbox_id * 7 + 3] * image_spacing[2] + image_origin[2];
		mask_roi->SetRegions(mask_roi_region);
		mask_roi->SetSpacing(mask_roi_spacing);
		mask_roi->SetOrigin(mask_roi_origin);
		mask_roi->Allocate();

		int mask_roi_buffer_length = bbox_size[0] * bbox_size[1] * bbox_size[2];
		memcpy(mask_roi->GetBufferPointer(), bbox_data + (bbox_id * 2 + 1) * mask_roi_buffer_length, mask_roi_buffer_length * sizeof(Dtype));

		typedef itk::IdentityTransform<double, 3> IdentityTransformType;
		typedef itk::ResampleImageFilter<MaskType, MaskType> ResampleImageFilterType;
		typedef itk::NearestNeighborInterpolateImageFunction<MaskType, double> InterpolatorType;

		//ResampleImageFilterType::Pointer resampler = ResampleImageFilterType::New();
		//resampler->SetInput(mask_roi);
		//resampler->SetSize(mask_size);
		//resampler->SetOutputSpacing(mask_spacing);
		//resampler->SetOutputOrigin(mask_origin);
		//resampler->SetTransform(IdentityTransformType::New());
		//resampler->Update();
		//MaskType::Pointer mask = resampler->GetOutput();

		ResampleImageFilterType::Pointer resampler = ResampleImageFilterType::New();
		resampler->SetInput(mask_roi);
		resampler->SetSize(output_size);
		resampler->SetOutputSpacing(output_spacing);
		resampler->SetOutputOrigin(output_origin);
		resampler->SetTransform(IdentityTransformType::New());
		resampler->Update();
		MaskType::Pointer mask = resampler->GetOutput();

		size_t pos = lines_[lines_id_].first.find_first_of('/');
		std::string file_name = lines_[lines_id_].first.substr(pos + 1);
		stringstream ss;
		ss << int(bbox_label[bbox_id] + 0.5);
		std::string output_path_name = rpn_output_param.roi_root_folder() + ss.str() + "/";
		std::string output_file_name = output_path_name + file_name + ".mhd";

		mkdir(output_path_name.c_str());

		typedef itk::ImageFileWriter<MaskType> WriterType;
		WriterType::Pointer writer = WriterType::New();
		writer->SetFileName(output_file_name);
		writer->SetInput(mask);
		writer->Update();
	}

	lines_id_ = (lines_id_ + 1) % lines_.size();
}

template <typename Dtype>
void RPNBBoxSegLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(RPNBBoxSegLayer);
REGISTER_LAYER_CLASS(RPNBBoxSeg);

}  // namespace caffe