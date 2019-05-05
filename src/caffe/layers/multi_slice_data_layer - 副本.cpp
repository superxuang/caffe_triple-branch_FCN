#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/multi_slice_data_layer.hpp"
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
#include <itkAffineTransform.h>
#include <itkCenteredSimilarity2DTransform.h>

using std::floor;

namespace caffe {

	template <typename Dtype>
	MultiSliceDataLayer<Dtype>::~MultiSliceDataLayer<Dtype>() {
		this->StopInternalThread();
		for (vector<SliceRecord*>::iterator it = slice_list_.begin(); it != slice_list_.end(); ++it) {
			if (NULL != *it) {
				delete *it;
				*it = NULL;
			}
		}
		slice_list_.clear();
		slice_id_list_.clear();
	}

	template <typename Dtype>
	void MultiSliceDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		output_labels_ = (top.size() > 1);

		itk::ObjectFactoryBase::RegisterFactory(itk::MetaImageIOFactory::New());
		typedef itk::ImageFileReader<ImageType> ImageReaderType;
		typedef itk::ImageFileReader<LabelType> LabelReaderType;
		typedef itk::ImageFileWriter<ImageSliceType> ImageSliceWriterType;

		const MHDDataParameter& mhd_data_param = this->layer_param_.mhd_data_param();
		const int batch_size = mhd_data_param.batch_size();
		const int num_of_slices = mhd_data_param.num_of_slices();
		ImageSliceType::RegionType slice_region;
		ImageSliceType::SizeType slice_size;
		ImageSliceType::SpacingType slice_spacing;
		const string& root_folder = mhd_data_param.root_folder();
		const ContourNameList& contour_name_list = mhd_data_param.contour_name_list();
		contour_num_ = contour_name_list.name_size();
		const string& source = mhd_data_param.source();
		LOG(INFO) << "Opening file " << source;

		std::ifstream infile(source.c_str());
		string line;
		size_t pos1, pos2;
		int file_num = 0;
		while (std::getline(infile, line)) {
			LOG(INFO) << "Loading file " << file_num << "...";
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
				for (int i = 0; i < contour_num_; ++i) {
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

			if (!contour_labels.empty()) {
				ImageType::DirectionType direct_src;
				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						direct_src[i][j] = (i == j) ? 1 : 0;
					}
				}

				ImageReaderType::Pointer reader_image = ImageReaderType::New();
				reader_image->SetFileName(root_folder + image_file_name);
				reader_image->Update();
				ImageType::Pointer image = reader_image->GetOutput();
				image->SetDirection(direct_src);

				// resample
				const ImageType::SizeType& origin_size = image->GetBufferedRegion().GetSize();
				const ImageType::SpacingType& origin_spacing = image->GetSpacing();
				const ImageType::PointType& origin_origin = image->GetOrigin();

				ImageType::SizeType resample_size;
				ImageType::SpacingType resample_spacing;
				ImageType::PointType resample_origin;

				resample_size[0] = mhd_data_param.resample_size_x();
				resample_size[1] = mhd_data_param.resample_size_y();
				resample_size[2] = mhd_data_param.resample_size_z();
				resample_spacing[0] = origin_size[0] * origin_spacing[0] / resample_size[0];
				resample_spacing[1] = origin_size[1] * origin_spacing[1] / resample_size[1];
				resample_spacing[2] = origin_size[2] * origin_spacing[2] / resample_size[2];
				resample_origin[0] = origin_origin[0];
				resample_origin[1] = origin_origin[1];
				resample_origin[2] = origin_origin[2];

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
				resampler_image->Update();
				image = resampler_image->GetOutput();

				// rescale intensity to [0, 1]
				const int image_buffer_length = image->GetLargestPossibleRegion().GetNumberOfPixels();
				VoxelType* image_buffer = image->GetBufferPointer();
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

				// extract slices
				switch (mhd_data_param.extract_direction())
				{
				case 0: // axial
				{
					for (int i = 0; i < resample_size[2]; ++i)
					{
						slice_size[0] = resample_size[0];
						slice_size[1] = resample_size[1];
						slice_region.SetSize(slice_size);
						slice_spacing[0] = resample_spacing[0];
						slice_spacing[1] = resample_spacing[1];
						ImageSliceType::Pointer slice = ImageSliceType::New();
						slice->SetRegions(slice_region);
						slice->SetSpacing(slice_spacing);
						slice->Allocate();
						VoxelType* slice_buffer = slice->GetBufferPointer();

						for (int j = 0; j < slice_size[1]; ++j)
						{
							for (int k = 0; k < slice_size[0]; ++k)
							{
								slice_buffer[j * slice_size[0] + k] = image_buffer[i * resample_size[1] * resample_size[0] + j * resample_size[0] + k];
							}
						}

						std::vector<int> labels;
						labels.resize(contour_num_, -1);
						for (int j = 0; j < exist_contours.size(); ++j)
						{
							int ii = i * resample_spacing[2] / origin_spacing[2];
							if (ii >= roi_z0[j] && ii <= roi_z1[j])
							{
								labels[exist_contours[j] - 1] = 1;
							}
							else
							{
								labels[exist_contours[j] - 1] = 0;
							}
						}

						SliceRecord* slice_record = new SliceRecord;
						slice_record->file_num_ = file_num;
						slice_record->data_ = slice;
						slice_record->label_.swap(labels);
						slice_list_.push_back(slice_record);
						slice_id_list_.push_back(slice_list_.size() - 1);
					}
					break;
				}
				case 1: // coronal
				{
					for (int i = 0; i < resample_size[1]; ++i)
					{
						slice_size[0] = resample_size[0];
						slice_size[1] = resample_size[2];
						slice_region.SetSize(slice_size);
						slice_spacing[0] = resample_spacing[0];
						slice_spacing[1] = resample_spacing[2];
						ImageSliceType::Pointer slice = ImageSliceType::New();
						slice->SetRegions(slice_region);
						slice->SetSpacing(slice_spacing);
						slice->Allocate();
						VoxelType* slice_buffer = slice->GetBufferPointer();

						for (int j = 0; j < slice_size[1]; ++j)
						{
							for (int k = 0; k < slice_size[0]; ++k)
							{
								slice_buffer[j * slice_size[0] + k] = image_buffer[j * resample_size[1] * resample_size[0] + i * resample_size[0] + k];
							}
						}

						std::vector<int> labels;
						labels.resize(contour_num_, -1);
						for (int j = 0; j < exist_contours.size(); ++j)
						{
							int ii = i * resample_spacing[1] / origin_spacing[1];
							if (ii >= roi_y0[j] && ii <= roi_y1[j])
							{
								labels[exist_contours[j] - 1] = 1;
							}
							else
							{
								labels[exist_contours[j] - 1] = 0;
							}
						}

						SliceRecord* slice_record = new SliceRecord;
						slice_record->file_num_ = file_num;
						slice_record->data_ = slice;
						slice_record->label_.swap(labels);
						slice_list_.push_back(slice_record);
						slice_id_list_.push_back(slice_list_.size() - 1);
					}
					break;
				}
				case 2: // sagittal
				{
					for (int i = 0; i < resample_size[0]; ++i)
					{
						slice_size[0] = resample_size[1];
						slice_size[1] = resample_size[2];
						slice_region.SetSize(slice_size);
						slice_spacing[0] = resample_spacing[1];
						slice_spacing[1] = resample_spacing[2];
						ImageSliceType::Pointer slice = ImageSliceType::New();
						slice->SetRegions(slice_region);
						slice->SetSpacing(slice_spacing);
						slice->Allocate();
						VoxelType* slice_buffer = slice->GetBufferPointer();

						for (int j = 0; j < slice_size[1]; ++j)
						{
							for (int k = 0; k < slice_size[0]; ++k)
							{
								slice_buffer[j * slice_size[0] + k] = image_buffer[j * resample_size[1] * resample_size[0] + k * resample_size[0] + i];
							}
						}

						std::vector<int> labels;
						labels.resize(contour_num_, -1);
						for (int j = 0; j < exist_contours.size(); ++j)
						{
							int ii = i * resample_spacing[0] / origin_spacing[0];
							if (ii >= roi_x0[j] && ii <= roi_x1[j])
							{
								labels[exist_contours[j] - 1] = 1;
							}
							else
							{
								labels[exist_contours[j] - 1] = 0;
							}
						}

						SliceRecord* slice_record = new SliceRecord;
						slice_record->file_num_ = file_num;
						slice_record->data_ = slice;
						slice_record->label_.swap(labels);
						slice_list_.push_back(slice_record);
						slice_id_list_.push_back(slice_list_.size() - 1);
					}
					break;
				}
				}
			}
			file_num++;
		}

		CHECK(!slice_list_.empty()) << "File is empty";

		if (this->phase_ == TRAIN && mhd_data_param.shuffle()) {
			// randomly shuffle data
			LOG(INFO) << "Shuffling data";
			const unsigned int prefetch_rng_seed = caffe_rng_rand();
			prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
			ShuffleImages();
		}

		int iters_per_epoch = 0;
		if (slice_list_.size() % batch_size == 0)
			iters_per_epoch = slice_list_.size() / batch_size;
		else
			iters_per_epoch = slice_list_.size() / batch_size + 1;
		LOG(INFO) << "A total of " << slice_list_.size() << " slices, " << iters_per_epoch << " iterations per epoch.";

		slice_id_ = 0;

		vector<int> data_shape(4);
		data_shape[0] = 1;
		data_shape[1] = num_of_slices * 2 + 1;
		data_shape[2] = slice_size[1];
		data_shape[3] = slice_size[0];
		vector<int> label_shape(3);
		if (this->phase_ == TRAIN)
		{
			label_shape[0] = 1;
			label_shape[1] = 1;
			label_shape[2] = contour_num_;
		}
		else
		{
			label_shape[0] = 1;
			label_shape[1] = 1;
			label_shape[2] = 1;
		}
		for (int i = 0; i < PREFETCH_COUNT; ++i) {
			this->prefetch_[i].data_.Reshape(data_shape);
			if (output_labels_)	{
				this->prefetch_[i].label_.Reshape(label_shape);
			}
		}
		top[0]->Reshape(data_shape);
		if (output_labels_)	{
			top[1]->Reshape(label_shape);
		}
	}

	template <typename Dtype>
	void MultiSliceDataLayer<Dtype>::ShuffleImages() {
		caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		shuffle(slice_id_list_.begin(), slice_id_list_.end(), prefetch_rng);
	}

	// This function is called on prefetch thread
	template <typename Dtype>
	void MultiSliceDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
		typedef itk::ImageFileReader<ImageType> ImageReaderType;
		typedef itk::ImageFileReader<LabelType> LabelReaderType;
		typedef itk::ImageFileWriter<ImageType> ImageWriterType;
		typedef itk::ImageFileWriter<LabelType> LabelWriterType;
		typedef itk::ImageFileReader<ImageSliceType> ImageSliceReaderType;
		typedef itk::ImageFileReader<LabelSliceType> LabelSliceReaderType;
		typedef itk::ImageFileWriter<ImageSliceType> ImageSliceWriterType;
		typedef itk::ImageFileWriter<LabelSliceType> LabelSliceWriterType;
		typedef itk::HistogramMatchingImageFilter<ImageType, ImageType> HMFilterType;
		typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
		typedef itk::ResampleImageFilter<LabelType, LabelType> ResampleLabelFilterType;
		typedef itk::NearestNeighborInterpolateImageFunction<LabelType, double> InterpolatorType;
		typedef itk::ResampleImageFilter<ImageSliceType, ImageSliceType> ResampleSliceFilterType;
		typedef itk::IdentityTransform<double, 2> IdentityTransformType;
		typedef itk::AffineTransform<double, 2> RotateTransformType;
		typedef itk::CenteredSimilarity2DTransform<double> TransformType;

		const int slice_num = slice_list_.size();
		CHECK_GT(slice_num, slice_id_);
		SliceRecord* slice_record = slice_list_[slice_id_list_[slice_id_]];
		const ImageSliceType::SizeType& slice_size = slice_record->data_->GetBufferedRegion().GetSize();
		const MHDDataParameter& mhd_data_param = this->layer_param_.mhd_data_param();
		const int batch_size = min((int)mhd_data_param.batch_size(), slice_num - slice_id_);
		const int num_of_slices = mhd_data_param.num_of_slices();
		const int spacing_between_slices = mhd_data_param.spacing_between_slices();
		const string& root_folder = mhd_data_param.root_folder();

		vector<int> data_shape(4);
		data_shape[0] = batch_size;
		data_shape[1] = num_of_slices * 2 + 1;
		data_shape[2] = slice_size[1];
		data_shape[3] = slice_size[0];
		batch->data_.Reshape(data_shape);
		Dtype* batch_data = batch->data_.mutable_cpu_data();
		memset(batch_data, 0, sizeof(Dtype) * batch->data_.count());

		Dtype* label_data = NULL;
		if (output_labels_)
		{
			vector<int> label_shape(3);
			if (this->phase_ == TRAIN)
			{
				label_shape[0] = batch_size;
				label_shape[1] = 1;
				label_shape[2] = contour_num_;
			}
			else
			{
				label_shape[0] = batch_size;
				label_shape[1] = 1;
				label_shape[2] = 1;
			}
			batch->label_.Reshape(label_shape);
			label_data = batch->label_.mutable_cpu_data();
			memset(label_data, 0, sizeof(Dtype) * batch->label_.count());
		}

		for (int batch_id = 0; batch_id < batch_size; ++batch_id)
		{
			SliceRecord* center_slice_record = slice_list_[slice_id_list_[slice_id_]];
			const int file_num = center_slice_record->file_num_;
			const ImageSliceType::SpacingType& slice_spacing = center_slice_record->data_->GetSpacing();

			RotateTransformType::Pointer rotate_transform = RotateTransformType::New();
			RotateTransformType::OutputVectorType translation;
			translation[0] = -(slice_spacing[0] * slice_size[0] * 0.5);
			translation[1] = -(slice_spacing[1] * slice_size[1] * 0.5);
			rotate_transform->Translate(translation);
			const unsigned int r = caffe_rng_rand();
			int angle_in_degree = r % 11 - 5;
			double angle = angle_in_degree * 3.141592654 / 180.0;
			rotate_transform->Rotate2D(angle, false);
			translation[0] = -translation[0];
			translation[1] = -translation[1];
			rotate_transform->Translate(translation, false);

			for (int i = -num_of_slices; i <= num_of_slices; ++i)
			{
				int ii = slice_id_list_[slice_id_] + i * spacing_between_slices;
				if ( ii < 0 || ii >= slice_num)
					continue;
				SliceRecord* slice_record = slice_list_[ii];
				if (slice_record->file_num_ != file_num)
					continue;
				ImageSliceType::Pointer slice = slice_record->data_;

				const ImageSliceType::PointType& slice_origin = slice->GetOrigin();

				ImageSliceType::SizeType resample_size = slice_size;
				ImageSliceType::SpacingType resample_spacing = slice_spacing;
				ImageSliceType::PointType resample_origin = slice_origin;

				ResampleSliceFilterType::Pointer resampler = ResampleSliceFilterType::New();
				resampler->SetInput(slice);
				resampler->SetSize(resample_size);
				resampler->SetOutputSpacing(resample_spacing);
				resampler->SetOutputOrigin(resample_origin);

				TransformType::Pointer transform = TransformType::New();
				TransformType::InputPointType rotation_center;
				rotation_center[0] = slice_origin[0] + slice_spacing[0] * slice_size[0] / 2.0;
				rotation_center[1] = slice_origin[1] + slice_spacing[1] * slice_size[1] / 2.0;
				transform->SetCenter(rotation_center);

				const int random_seed_resolution = 1000;
				int random_seed = caffe_rng_rand() % (2 * random_seed_resolution + 1) - random_seed_resolution;
				double angle_in_degrees = mhd_data_param.augment_angle() * random_seed / random_seed_resolution;
				double angle_in_radians = angle_in_degrees * 3.141592654 / 180.0;
				transform->SetAngle(angle_in_radians);

				random_seed = caffe_rng_rand() % (2 * random_seed_resolution + 1) - random_seed_resolution;
				double scale = 1.0 + mhd_data_param.augment_scale() * random_seed / random_seed_resolution;
				transform->SetScale(scale);

				TransformType::OutputVectorType translation;
				random_seed = caffe_rng_rand() % (2 * random_seed_resolution + 1) - random_seed_resolution;
				translation[0] = slice_spacing[0] * slice_size[0] * mhd_data_param.augment_translate() * random_seed / random_seed_resolution;
				random_seed = caffe_rng_rand() % (2 * random_seed_resolution + 1) - random_seed_resolution;
				translation[1] = slice_spacing[1] * slice_size[1] * mhd_data_param.augment_translate() * random_seed / random_seed_resolution;
				transform->SetTranslation(translation);

				if (this->phase_ == TRAIN)
				{
					resampler->SetTransform(transform);
				}
				else
				{
					resampler->SetTransform(IdentityTransformType::New());
				}
				resampler->Update();
				ImageSliceType::Pointer resample_slice = resampler->GetOutput();

				//ImageSliceWriterType::Pointer slice_writer = ImageSliceWriterType::New();
				//slice_writer->SetFileName("F:/slice_origin.mhd");
				//slice_writer->SetInput(slice);
				//slice_writer->Update();
				//slice_writer->SetFileName("F:/slice_augment.mhd");
				//slice_writer->SetInput(resample_slice);
				//slice_writer->Update();

				VoxelType* slice_buffer = resample_slice->GetBufferPointer();
				int batch_data_start_ind = batch_id * (2 * num_of_slices + 1) * slice_size[1] * slice_size[0] + (i + num_of_slices) * slice_size[1] * slice_size[0];
				for (int j = 0; j < slice_size[1] * slice_size[0]; ++j)
				{
					batch_data[batch_data_start_ind + j] = (Dtype)(slice_buffer[j]);
				}
				
				if (i == 0 && output_labels_)
				{
					if (this->phase_ == TRAIN)
					{
						for (int j = 0; j < contour_num_; ++j)
						{
							label_data[batch_id * contour_num_ + j] = center_slice_record->label_[j];
						}
					}
					else
					{
						label_data[batch_id] = file_num;
					}
				}
			}

			slice_id_++;
		}

		if (slice_id_ >= slice_num) {
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			slice_id_ = 0;
			if (this->phase_ == TRAIN && mhd_data_param.shuffle()) {
				ShuffleImages();
			}
		}
	}

	INSTANTIATE_CLASS(MultiSliceDataLayer);
	REGISTER_LAYER_CLASS(MultiSliceData);

}  // namespace caffe
