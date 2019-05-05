#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/mhd_roi_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <itkMetaImageIOFactory.h>
#include <itkIdentityTransform.h>
#include <itkResampleImageFilter.h>
#include <itkBSplineTransform.h>
#include <itkBSplineTransformInitializer.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkHistogramMatchingImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkSobelEdgeDetectionImageFilter.h>

using std::floor;

namespace caffe {

template <typename Dtype>
MHDRoiDataLayer<Dtype>::~MHDRoiDataLayer<Dtype>() {
  this->StopInternalThread();
  for (vector<ImageRecord*>::iterator it = lines_.begin(); it != lines_.end(); ++it) {
    if (NULL != *it) {
      delete *it; 
      *it = NULL;
    }
  }
  lines_.clear();
}

template <typename Dtype>
void MHDRoiDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	itk::ObjectFactoryBase::RegisterFactory(itk::MetaImageIOFactory::New());
	typedef itk::ImageFileReader<ImageType> ImageReaderType;
	typedef itk::ImageFileReader<LabelType> LabelReaderType;

	const MHDDataParameter& mhd_data_param = this->layer_param_.mhd_data_param();
	fixed_spacing_ = 
		mhd_data_param.has_resample_spacing_x() && 
		mhd_data_param.has_resample_spacing_y() && 
		mhd_data_param.has_resample_spacing_z();
	const string& root_folder = mhd_data_param.root_folder();
	const ContourNameList& contour_name_list = mhd_data_param.contour_name_list();
	const int class_num = contour_name_list.name_size();
	const string& source = mhd_data_param.source();
	LOG(INFO) << "Opening file " << source;

	std::ifstream infile(source.c_str());
	string line;
	size_t pos1, pos2;
	int file_num = 0;
	while (std::getline(infile, line)) {
		LOG(INFO) << "Loading file " << file_num++ << "...";
		pos1 = line.find_first_of(' ');
		pos2 = line.find_last_of(' ');
		string image_file_name = line.substr(0, pos1);
		string label_file_name = line.substr(pos1 + 1, pos2 - pos1 - 1);
		string info_file_name = line.substr(pos2 + 1);
		std::ifstream infile_info(root_folder + info_file_name);
		std::vector<int> contour_labels;
		std::vector<int> roi_x0, roi_x1, roi_y0, roi_y1, roi_z0, roi_z1;
		std::vector<int> exist_contours;
		int label_range[3][2];
		label_range[0][0] = INT_MAX;
		label_range[0][1] = INT_MIN;
		label_range[1][0] = INT_MAX;
		label_range[1][1] = INT_MIN;
		label_range[2][0] = INT_MAX;
		label_range[2][1] = INT_MIN;
		while (std::getline(infile_info, line)) {
			pos1 = string::npos;
			for (int i = 0; i < contour_name_list.name_size(); ++i) {
				pos1 = line.find(contour_name_list.name(i));
				if (pos1 != string::npos) {
					exist_contours.push_back(i + 1);
					break;
				}
			}
			if (pos1 == string::npos)
				continue;

			pos1 = line.find_first_of(' ', pos1);
			pos2 = line.find_first_of(' ', pos1 + 1);
			int label_value = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
			contour_labels.push_back(label_value);

			pos1 = pos2;
			pos2 = line.find_first_of(' ', pos1 + 1);
			int x0 = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
			roi_x0.push_back(x0);
			label_range[0][0] = min(label_range[0][0], x0);

			pos1 = pos2;
			pos2 = line.find_first_of(' ', pos1 + 1);
			int x1 = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
			roi_x1.push_back(x1);
			label_range[0][1] = max(label_range[0][1], x1);

			pos1 = pos2;
			pos2 = line.find_first_of(' ', pos1 + 1);
			int y0 = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
			roi_y0.push_back(y0);
			label_range[1][0] = min(label_range[1][0], y0);

			pos1 = pos2;
			pos2 = line.find_first_of(' ', pos1 + 1);
			int y1 = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
			roi_y1.push_back(y1);
			label_range[1][1] = max(label_range[1][1], y1);

			pos1 = pos2;
			pos2 = line.find_first_of(' ', pos1 + 1);
			int z0 = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
			roi_z0.push_back(z0);
			label_range[2][0] = min(label_range[2][0], z0);

			pos1 = pos2;
			int z1 = atoi(line.substr(pos1 + 1).c_str());
			roi_z1.push_back(z1);
			label_range[2][1] = max(label_range[2][1], z1);
		}
		if (!contour_labels.empty() || this->phase() == TEST) {
			ImageType::DirectionType direct_src;
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					direct_src[i][j] = (i == j) ? 1 : 0;
				}
			}

			ImageRecord* image_record = new ImageRecord;
			image_record->roi_label_.swap(exist_contours);
			image_record->roi_x0_.swap(roi_x0);
			image_record->roi_x1_.swap(roi_x1);
			image_record->roi_y0_.swap(roi_y0);
			image_record->roi_y1_.swap(roi_y1);
			image_record->roi_z0_.swap(roi_z0);
			image_record->roi_z1_.swap(roi_z1);

			ImageReaderType::Pointer reader_image = ImageReaderType::New();
			reader_image->SetFileName(root_folder + image_file_name);
			reader_image->Update();
			ImageType::Pointer image = reader_image->GetOutput();
			image->SetDirection(direct_src);

			// resample
			{
				const ImageType::SizeType& origin_size = image->GetBufferedRegion().GetSize();
				const ImageType::SpacingType& origin_spacing = image->GetSpacing();
				const ImageType::PointType& origin_origin = image->GetOrigin();

				ImageType::SizeType resample_size;
				ImageType::SpacingType resample_spacing;
				ImageType::PointType resample_origin;

				if (fixed_spacing_)
				{
					resample_spacing[0] = mhd_data_param.resample_spacing_x();
					resample_spacing[1] = mhd_data_param.resample_spacing_y();
					resample_spacing[2] = mhd_data_param.resample_spacing_z();
					resample_size[0] = mhd_data_param.resample_size_x();
					resample_size[1] = mhd_data_param.resample_size_y();
					resample_size[2] = origin_size[2] * origin_spacing[2] / resample_spacing[2];
				}
				else
				{
					resample_size[0] = mhd_data_param.resample_size_x();
					resample_size[1] = mhd_data_param.resample_size_y();
					resample_size[2] = mhd_data_param.resample_size_z();
					resample_spacing[0] = origin_size[0] * origin_spacing[0] / resample_size[0];
					resample_spacing[1] = origin_size[1] * origin_spacing[1] / resample_size[1];
					resample_spacing[2] = origin_size[2] * origin_spacing[2] / resample_size[2];
				}
				resample_origin[0] = origin_origin[0] + origin_size[0] * origin_spacing[0] * 0.5 - resample_size[0] * resample_spacing[0] * 0.5;
				resample_origin[1] = origin_origin[1] + origin_size[1] * origin_spacing[1] * 0.5 - resample_size[1] * resample_spacing[1] * 0.5;
				resample_origin[2] = origin_origin[2] + origin_size[2] * origin_spacing[2] * 0.5 - resample_size[2] * resample_spacing[2] * 0.5;

				image_record->size_[0] = mhd_data_param.resample_size_x();
				image_record->size_[1] = mhd_data_param.resample_size_y();
				image_record->size_[2] = mhd_data_param.resample_size_z();
				for (int i = 0; i < 3; ++i)	{
					image_record->origin_size_[i] = origin_size[i];
					image_record->origin_spacing_[i] = origin_spacing[i];
					image_record->origin_origin_[i] = origin_origin[i];
					image_record->spacing_[i] = resample_spacing[i];
					if (fixed_spacing_)
					{
						image_record->origin_[i] = origin_origin[i] + origin_size[i] * origin_spacing[i] * 0.5 - image_record->size_[i] * image_record->spacing_[i] * 0.5;

					}
					else
					{
						image_record->origin_[i] = resample_origin[i];
					}
					for (int j = 0; j < 2; ++j) {
						image_record->label_range_[i][j] = label_range[i][j];
					}
				}

				typedef itk::IdentityTransform<double, 3> IdentityTransformType;
				typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
				typedef itk::ResampleImageFilter<LabelType, LabelType> ResampleLabelFilterType;
				typedef itk::NearestNeighborInterpolateImageFunction<LabelType, double> InterpolatorType;

				ResampleImageFilterType::Pointer resampler_image = ResampleImageFilterType::New();
				resampler_image->SetInput(image);
				resampler_image->SetSize(resample_size);
				resampler_image->SetOutputSpacing(resample_spacing);
				resampler_image->SetOutputOrigin(resample_origin);
				resampler_image->SetTransform(IdentityTransformType::New());
				resampler_image->SetDefaultPixelValue(-2000);
				resampler_image->Update();
				image = resampler_image->GetOutput();

			}
			const int image_buffer_length = image->GetBufferedRegion().GetNumberOfPixels();
			Dtype* image_buffer = image->GetBufferPointer();

			// rescale intensity to [0, 1]
			const Dtype min_intensity = mhd_data_param.min_intensity();
			const Dtype max_intensity = mhd_data_param.max_intensity();
			for (int i = 0; i < image_buffer_length; ++i) {
				if (image_buffer[i] < min_intensity)
					image_buffer[i] = min_intensity;
				if (image_buffer[i] > max_intensity)
					image_buffer[i] = max_intensity;
			}
			double buff_scale = (max_intensity > min_intensity) ? 1.0 / (max_intensity - min_intensity) : 0.0;
			for (int i = 0; i < image_buffer_length; ++i) {
				image_buffer[i] = (image_buffer[i] - min_intensity) * buff_scale;
			}

#ifdef USE_MULTI_CHANNEL
			// enhanced image
			ImageType::Pointer enhanced_image = ImageType::New();
			enhanced_image->SetRegions(image->GetBufferedRegion());
			enhanced_image->SetSpacing(image->GetSpacing());
			enhanced_image->SetOrigin(image->GetOrigin());
			enhanced_image->Allocate();
			Dtype* enhanced_image_buffer = enhanced_image->GetBufferPointer();
			Dtype wl = ((Dtype)100 - min_intensity) / (max_intensity - min_intensity);
			Dtype ww = (Dtype)150 / (max_intensity - min_intensity);
			for (int i = 0; i < image_buffer_length; ++i)
			{
				Dtype v = (image_buffer[i] - wl) / ww;
				enhanced_image_buffer[i] = exp(-v * v);
			}

			// edge image
			typedef itk::SobelEdgeDetectionImageFilter<ImageType, ImageType> EdgeFilterType;
			EdgeFilterType::Pointer sobelFilter = EdgeFilterType::New();
			sobelFilter->SetInput(enhanced_image);
			sobelFilter->Update();
			ImageType::Pointer edge_image = sobelFilter->GetOutput();
			Dtype* edge_image_buffer = edge_image->GetBufferPointer();
			for (int i = 0; i < image_buffer_length; ++i)
			{
				Dtype v = edge_image_buffer[i] / 10.0;
				edge_image_buffer[i] = (v > 1.0) ? 1.0 : v;
			}
#endif

			image_record->image_ = image;
#ifdef USE_MULTI_CHANNEL
			image_record->enhanced_image_ = enhanced_image;
			image_record->edge_image_ = edge_image;
#endif
			image_record->info_file_name_ = info_file_name;
			lines_.push_back(image_record);
		}
	}

	CHECK(!lines_.empty()) << "File is empty";

	if (this->phase_ == TRAIN && mhd_data_param.shuffle()) {
		// randomly shuffle data
		LOG(INFO) << "Shuffling data";
		const unsigned int prefetch_rng_seed = caffe_rng_rand();
		prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
		ShuffleImages();
	} else {
		if (this->phase_ == TRAIN &&
			mhd_data_param.rand_skip() == 0) {
			LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
		}
	}
	LOG(INFO) << "A total of " << lines_.size() << " images.";

	lines_id_ = 0;
	// Check if we would need to randomly skip a few data points
	if (mhd_data_param.rand_skip()) {
		unsigned int skip = caffe_rng_rand() % mhd_data_param.rand_skip();
		LOG(INFO) << "Skipping first " << skip << " data points.";
		CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
		lines_id_ = skip;
	}

	const unsigned int ind_rng_seed = caffe_rng_rand();
	ind_rng_.reset(new Caffe::RNG(ind_rng_seed));
	const unsigned int trans_rng_seed = caffe_rng_rand();
	trans_rng_.reset(new Caffe::RNG(trans_rng_seed));

	vector<int> data_shape(5);
	data_shape[0] = 1;
#ifdef USE_MULTI_CHANNEL
	data_shape[1] = 3;
#else
	data_shape[1] = 1;
#endif
	data_shape[2] = mhd_data_param.resample_size_z();
	data_shape[3] = mhd_data_param.resample_size_y();
	data_shape[4] = mhd_data_param.resample_size_x();
	vector<int> info_shape(2);
	info_shape[0] = 1;
	info_shape[1] = 19;
	vector<int> label_a_shape(3);
	label_a_shape[0] = 1;
	label_a_shape[1] = 1;
	label_a_shape[2] = class_num * mhd_data_param.resample_size_z();
	vector<int> label_c_shape(3);
	label_c_shape[0] = 1;
	label_c_shape[1] = 1;
	label_c_shape[2] = class_num * mhd_data_param.resample_size_y();
	vector<int> label_s_shape(3);
	label_s_shape[0] = 1;
	label_s_shape[1] = 1;
	label_s_shape[2] = class_num * mhd_data_param.resample_size_x();
	for (int i = 0; i < PREFETCH_COUNT; ++i) {
		this->prefetch_[i].data_.Reshape(data_shape);
		this->prefetch_[i].info_.Reshape(info_shape);
		if (output_labels_)
		{
			this->prefetch_[i].label_a_.Reshape(label_a_shape);
			this->prefetch_[i].label_c_.Reshape(label_c_shape);
			this->prefetch_[i].label_s_.Reshape(label_s_shape);
		}
	}

	top[0]->Reshape(data_shape);
	top[1]->Reshape(info_shape);
	if (output_labels_)
	{
		top[2]->Reshape(label_a_shape);
		top[3]->Reshape(label_c_shape);
		top[4]->Reshape(label_s_shape);
	}
}

template <typename Dtype>
void MHDRoiDataLayer<Dtype>::ShuffleImages() {
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void MHDRoiDataLayer<Dtype>::load_batch(RoiBatch<Dtype>* batch) {

	typedef itk::ImageFileReader<ImageType> ImageReaderType;
	typedef itk::ImageFileReader<LabelType> LabelReaderType;
	typedef itk::ImageFileWriter<ImageType> ImageWriterType;
	typedef itk::ImageFileWriter<LabelType> LabelWriterType;
	typedef itk::HistogramMatchingImageFilter<ImageType, ImageType> HMFilterType;
	typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
	typedef itk::ResampleImageFilter<LabelType, LabelType> ResampleLabelFilterType;
	typedef itk::NearestNeighborInterpolateImageFunction<LabelType, double> InterpolatorType;

	const MHDDataParameter& mhd_data_param = this->layer_param_.mhd_data_param();
	const ContourNameList& contour_name_list = mhd_data_param.contour_name_list();
	const int class_num = contour_name_list.name_size();
	const int lines_size = lines_.size();

	const int batch_size = (this->phase_ == TEST) ? std::min((int)mhd_data_param.batch_size(), lines_size - lines_id_) : mhd_data_param.batch_size();

	const int min_image_length = mhd_data_param.min_truncate_length();
	double crop_prob;
	bool need_crop;
	int crop_length;
	int crop_start;

	for (int batch_id = 0; batch_id < batch_size; ++batch_id) 
	{
		CHECK_GT(lines_size, lines_id_);
		const ImageRecord* image_record = lines_[lines_id_];
		ImageType::Pointer src_image = image_record->image_;
#ifdef USE_MULTI_CHANNEL
		ImageType::Pointer src_enhanced_image = image_record->enhanced_image_;
		ImageType::Pointer src_edge_image = image_record->edge_image_;
#endif
		const ImageType::RegionType& src_image_region = src_image->GetBufferedRegion();
		const ImageType::SizeType& src_image_size = src_image->GetBufferedRegion().GetSize();
		const ImageType::SpacingType& src_image_spacing = src_image->GetSpacing();
		const ImageType::PointType& src_image_origin = src_image->GetOrigin();
		const ImageType::DirectionType& src_image_direct = src_image->GetDirection();
		const int roi_num = image_record->roi_label_.size();

		if (batch_id == 0)
		{
			vector<int> data_shape(5);
			data_shape[0] = batch_size;
#ifdef USE_MULTI_CHANNEL
			data_shape[1] = 3;
#else
			data_shape[1] = 1;
#endif
			data_shape[2] = mhd_data_param.resample_size_z();
			data_shape[3] = mhd_data_param.resample_size_y();
			data_shape[4] = mhd_data_param.resample_size_x();
			batch->data_.Reshape(data_shape);

			std::vector<int> data_info_shape(2);
			data_info_shape[0] = batch_size;
			data_info_shape[1] = 19;
			batch->info_.Reshape(data_info_shape);

			vector<int> label_a_shape(3);
			label_a_shape[0] = batch_size;
			label_a_shape[1] = 1;
			label_a_shape[2] = class_num * mhd_data_param.resample_size_z();
			vector<int> label_c_shape(3);
			label_c_shape[0] = batch_size;
			label_c_shape[1] = 1;
			label_c_shape[2] = class_num * mhd_data_param.resample_size_y();
			vector<int> label_s_shape(3);
			label_s_shape[0] = batch_size;
			label_s_shape[1] = 1;
			label_s_shape[2] = class_num * mhd_data_param.resample_size_x();

			if (output_labels_) {
				batch->label_a_.Reshape(label_a_shape);
				batch->label_c_.Reshape(label_c_shape);
				batch->label_s_.Reshape(label_s_shape);
			}
		}

		ImageType::RegionType image_region = src_image_region;
		ImageType::SizeType image_size = src_image_size;
		ImageType::SpacingType image_spacing = src_image_spacing;
		ImageType::PointType image_origin = src_image_origin;
		ImageType::DirectionType image_direct = src_image_direct;

		double z_min = image_origin[2];
		double z_max = image_origin[2] + image_size[2] * image_spacing[2];

		crop_prob = (caffe_rng_rand() % 100) / 99.0;
		need_crop = (this->phase_ == TRAIN && crop_prob < mhd_data_param.truncate_probability() && src_image_size[2] > min_image_length);
		if (need_crop) {
			crop_length = (caffe_rng_rand() % (src_image_size[2] - min_image_length)) + min_image_length;
			crop_length = min(crop_length, (int)mhd_data_param.resample_size_z());
			crop_start = caffe_rng_rand() % (src_image_size[2] - crop_length + 1);
			if (crop_length * src_image_spacing[2] < 300)
			{
				crop_length = src_image_size[2];
				crop_start = 0;
			}
			image_origin[2] = src_image_origin[2] + crop_start * src_image_spacing[2];
			if (fixed_spacing_)
			{
				image_size[2] = crop_length;
			}
			else
			{
				image_spacing[2] = crop_length * src_image_spacing[2] / src_image_size[2];
			}
			z_min = image_origin[2];
			z_max = image_origin[2] + image_size[2] * image_spacing[2];
		}
		else
		{
			if (image_size[2] > mhd_data_param.resample_size_z())
			{
				z_min = src_image_origin[2] + src_image_size[2] * src_image_spacing[2] * 0.5 - mhd_data_param.resample_size_z() * src_image_spacing[2] * 0.5;
				z_max = z_min + mhd_data_param.resample_size_z() * image_spacing[2];
			}
			image_size[2] = mhd_data_param.resample_size_z();
			image_origin[2] = src_image_origin[2] + src_image_size[2] * src_image_spacing[2] * 0.5 - image_size[2] * src_image_spacing[2] * 0.5;
		}

		const int inplane_shift = mhd_data_param.inplane_shift();
		if (this->phase_ == TRAIN && inplane_shift > 0) {
			const int shift_x = caffe_rng_rand() % (inplane_shift * 2 + 1) - inplane_shift;
			const int shift_y = caffe_rng_rand() % (inplane_shift * 2 + 1) - inplane_shift;
			image_origin[0] = src_image_origin[0] + shift_x * src_image_spacing[0];
			image_origin[1] = src_image_origin[1] + shift_y * src_image_spacing[1];
		}
		
		typedef itk::IdentityTransform<double, 3> IdentityTransformType;
		typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
		typedef itk::ResampleImageFilter<LabelType, LabelType> ResampleLabelFilterType;
		typedef itk::NearestNeighborInterpolateImageFunction<LabelType, double> InterpolatorType;	

		//ImageWriterType::Pointer writer = ImageWriterType::New();
		//writer->SetInput(src_image);
		//writer->SetFileName("F:/step00.mhd");
		//writer->Update();
		//writer->SetInput(src_enhanced_image);
		//writer->SetFileName("F:/step01.mhd");
		//writer->Update();
		//writer->SetInput(src_edge_image);
		//writer->SetFileName("F:/step02.mhd");
		//writer->Update();

		ResampleImageFilterType::Pointer image_resampler = ResampleImageFilterType::New();
		image_resampler->SetInput(src_image);
		image_resampler->SetSize(image_size);
		image_resampler->SetOutputSpacing(image_spacing);
		image_resampler->SetOutputOrigin(image_origin);
		image_resampler->SetTransform(IdentityTransformType::New());
		image_resampler->SetDefaultPixelValue(0);
		image_resampler->Update();
		ImageType::Pointer resampled_image = image_resampler->GetOutput();
		resampled_image->DisconnectPipeline();

#ifdef USE_MULTI_CHANNEL
		image_resampler->SetInput(src_enhanced_image);
		image_resampler->SetSize(image_size);
		image_resampler->SetOutputSpacing(image_spacing);
		image_resampler->SetOutputOrigin(image_origin);
		image_resampler->SetTransform(IdentityTransformType::New());
		image_resampler->SetDefaultPixelValue(0);
		image_resampler->Update();
		ImageType::Pointer resampled_enhanced_image = image_resampler->GetOutput();
		resampled_enhanced_image->DisconnectPipeline();

		image_resampler->SetInput(src_edge_image);
		image_resampler->SetSize(image_size);
		image_resampler->SetOutputSpacing(image_spacing);
		image_resampler->SetOutputOrigin(image_origin);
		image_resampler->SetTransform(IdentityTransformType::New());
		image_resampler->SetDefaultPixelValue(0);
		image_resampler->Update();
		ImageType::Pointer resampled_edge_image = image_resampler->GetOutput();
		resampled_edge_image->DisconnectPipeline();
#endif

		//writer->SetInput(resampled_image);
		//writer->SetFileName("F:/step10.mhd");
		//writer->Update();
		//writer->SetInput(resampled_enhanced_image);
		//writer->SetFileName("F:/step11.mhd");
		//writer->Update();
		//writer->SetInput(resampled_edge_image);
		//writer->SetFileName("F:/step12.mhd");
		//writer->Update();

		ImageType::Pointer image;
		ImageType::Pointer enhanced_image;
		ImageType::Pointer edge_image;
		if (image_size[2] == mhd_data_param.resample_size_z())
		{
			image = resampled_image;
#ifdef USE_MULTI_CHANNEL
			enhanced_image = resampled_enhanced_image;
			edge_image = resampled_edge_image;
#endif
		}
		else
		{
			const int z_offset = max(((int)mhd_data_param.resample_size_z() - (int)image_size[2]) / 2, 0);
			image_size[2] = mhd_data_param.resample_size_z();
			image_origin[2] = image_origin[2] - z_offset * image_spacing[2];
			image_region.SetSize(image_size);

			image = ImageType::New();
			image->SetRegions(image_region);
			image->SetSpacing(image_spacing);
			image->SetOrigin(image_origin);
			image->Allocate();
			memset(image->GetBufferPointer(), 0, sizeof(Dtype) * image->GetBufferedRegion().GetNumberOfPixels());
			memcpy(image->GetBufferPointer() + z_offset * image_size[0] * image_size[1], resampled_image->GetBufferPointer(), sizeof(Dtype) * resampled_image->GetBufferedRegion().GetNumberOfPixels());

#ifdef USE_MULTI_CHANNEL
			enhanced_image = ImageType::New();
			enhanced_image->SetRegions(image_region);
			enhanced_image->SetSpacing(image_spacing);
			enhanced_image->SetOrigin(image_origin);
			enhanced_image->Allocate();
			memset(enhanced_image->GetBufferPointer(), 0, sizeof(Dtype) * enhanced_image->GetBufferedRegion().GetNumberOfPixels());
			memcpy(enhanced_image->GetBufferPointer() + z_offset * image_size[0] * image_size[1], resampled_enhanced_image->GetBufferPointer(), sizeof(Dtype) * resampled_enhanced_image->GetBufferedRegion().GetNumberOfPixels());

			edge_image = ImageType::New();
			edge_image->SetRegions(image_region);
			edge_image->SetSpacing(image_spacing);
			edge_image->SetOrigin(image_origin);
			edge_image->Allocate();
			memset(edge_image->GetBufferPointer(), 0, sizeof(Dtype) * edge_image->GetBufferedRegion().GetNumberOfPixels());
			memcpy(edge_image->GetBufferPointer() + z_offset * image_size[0] * image_size[1], resampled_edge_image->GetBufferPointer(), sizeof(Dtype) * resampled_edge_image->GetBufferedRegion().GetNumberOfPixels());
#endif
		}
		//writer->SetInput(image);
		//writer->SetFileName("F:/step20.mhd");
		//writer->Update();
		//writer->SetInput(enhanced_image);
		//writer->SetFileName("F:/step21.mhd");
		//writer->Update();
		//writer->SetInput(edge_image);
		//writer->SetFileName("F:/step22.mhd");
		//writer->Update();

		Dtype* image_buffer = image->GetBufferPointer();
#ifdef USE_MULTI_CHANNEL
		Dtype* enhanced_image_buffer = enhanced_image->GetBufferPointer();
		Dtype* edge_image_buffer = edge_image->GetBufferPointer();
#endif

		const int pixel_num = image_size[0] * image_size[1] * image_size[2];
		const int slice_pixel_num = image_size[0] * image_size[1];

#ifndef USE_MULTI_CHANNEL
		Dtype* image_data = batch->data_.mutable_cpu_data() + batch_id * pixel_num;
#else
		Dtype* image_data = batch->data_.mutable_cpu_data() + batch_id * 3 * pixel_num;
		Dtype* enhanced_image_data = batch->data_.mutable_cpu_data() + batch_id * 3 * pixel_num + pixel_num;
		Dtype* edge_image_data = batch->data_.mutable_cpu_data() + batch_id * 3 * pixel_num + 2 * pixel_num;
#endif
		for (int img_index = 0; img_index < pixel_num; ++img_index) {
			image_data[img_index] = image_buffer[img_index];
#ifdef USE_MULTI_CHANNEL
			enhanced_image_data[img_index] = enhanced_image_buffer[img_index];
			edge_image_data[img_index] = edge_image_buffer[img_index];
#endif
		}

		Dtype* info_data = batch->info_.mutable_cpu_data() + batch_id * 19;
		info_data[0] = image_size[2];
		info_data[1] = image_size[1];
		info_data[2] = image_size[0];
		info_data[3] = image_spacing[2];
		info_data[4] = image_spacing[1];
		info_data[5] = image_spacing[0];
		info_data[6] = image_origin[2];
		info_data[7] = image_origin[1];
		info_data[8] = image_origin[0];
		info_data[9] = image_record->origin_size_[2];
		info_data[10] = image_record->origin_size_[1];
		info_data[11] = image_record->origin_size_[0];
		info_data[12] = image_record->origin_spacing_[2];
		info_data[13] = image_record->origin_spacing_[1];
		info_data[14] = image_record->origin_spacing_[0];
		info_data[15] = image_record->origin_origin_[2];
		info_data[16] = image_record->origin_origin_[1];
		info_data[17] = image_record->origin_origin_[0];
		info_data[18] = 1;

		if (output_labels_) {
			Dtype* label_a_data = batch->label_a_.mutable_cpu_data() + batch_id * class_num * image_size[2];
			Dtype* label_c_data = batch->label_c_.mutable_cpu_data() + batch_id * class_num * image_size[1];
			Dtype* label_s_data = batch->label_s_.mutable_cpu_data() + batch_id * class_num * image_size[0];
			//memset(label_a_data, 0, sizeof(Dtype) * class_num * image_size[2]);
			for (int i = 0; i < class_num * image_size[2]; ++i)
			{
				label_a_data[i] = (Dtype)-1.;
			}
			memset(label_c_data, 0, sizeof(Dtype) * class_num * image_size[1]);
			memset(label_s_data, 0, sizeof(Dtype) * class_num * image_size[0]);
			for (int roi_id = 0; roi_id < roi_num; ++roi_id)
			{
				const int cls = image_record->roi_label_[roi_id] - 1;
				const double x0 = image_record->roi_x0_[roi_id] * image_record->origin_spacing_[0] + image_record->origin_origin_[0];
				const double y0 = image_record->roi_y0_[roi_id] * image_record->origin_spacing_[1] + image_record->origin_origin_[1];
				const double z0 = image_record->roi_z0_[roi_id] * image_record->origin_spacing_[2] + image_record->origin_origin_[2];
				const double x1 = image_record->roi_x1_[roi_id] * image_record->origin_spacing_[0] + image_record->origin_origin_[0];
				const double y1 = image_record->roi_y1_[roi_id] * image_record->origin_spacing_[1] + image_record->origin_origin_[1];
				const double z1 = image_record->roi_z1_[roi_id] * image_record->origin_spacing_[2] + image_record->origin_origin_[2];

				if (z0 >= z_min && z1 <= z_max)
				{
					for (int i = 0; i < image_size[2]; ++i)
					{
						const double pos = i * image_spacing[2] + image_origin[2];
						if (pos >= z0 && pos <= z1)
						{
							label_a_data[cls * image_size[2] + i] = 1;
						}
						else
						{
							label_a_data[cls * image_size[2] + i] = 0;
						}
					}
					for (int i = 0; i < image_size[1]; ++i)
					{
						const double pos = i * image_spacing[1] + image_origin[1];
						if (pos >= y0 && pos <= y1)
						{
							label_c_data[cls * image_size[1] + i] = 1;
						}
						else
						{
							label_c_data[cls * image_size[1] + i] = 0;
						}
					}
					for (int i = 0; i < image_size[0]; ++i)
					{
						const double pos = i * image_spacing[0] + image_origin[0];
						if (pos >= x0 && pos <= x1)
						{
							label_s_data[cls * image_size[0] + i] = 1;
						}
						else
						{
							label_s_data[cls * image_size[0] + i] = 0;
						}
					}
				}
			}
		}
		lines_id_++;
		// go to the next iter
		if (lines_id_ >= lines_size) {
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if (this->phase_ == TRAIN && this->layer_param_.mhd_data_param().shuffle()) {
				ShuffleImages();
			}
		}
	}
}

INSTANTIATE_CLASS(MHDRoiDataLayer);
REGISTER_LAYER_CLASS(MHDRoiData);

}  // namespace caffe
