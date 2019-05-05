#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/mhd_slice_data_layer.hpp"
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
	MHDSliceDataLayer<Dtype>::~MHDSliceDataLayer<Dtype>() {
		this->StopInternalThread();
		for (vector<VolumeRecord*>::iterator it = volume_list_.begin(); it != volume_list_.end(); ++it) {
			if (NULL != *it) {
				delete *it;
				*it = NULL;
			}
		}
		volume_list_.clear();
		volume_id_list_.clear();
	}

	template <typename Dtype>
	void MHDSliceDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		output_labels_ = (top.size() > 1);

		itk::ObjectFactoryBase::RegisterFactory(itk::MetaImageIOFactory::New());
		typedef itk::ImageFileReader<ImageType> ImageReaderType;
		typedef itk::ImageFileReader<LabelType> LabelReaderType;
		typedef itk::ImageFileWriter<ImageSliceType> ImageSliceWriterType;

		const MHDDataParameter& mhd_data_param = this->layer_param_.mhd_data_param();
		const int batch_size = mhd_data_param.batch_size();
		const string& root_folder = mhd_data_param.root_folder();
		const ContourNameList& contour_name_list = mhd_data_param.contour_name_list();
		const ContourPairList& contour_pair_list = mhd_data_param.contour_pair_list();
		contour_num_ = contour_name_list.name_size();
		const string& source = mhd_data_param.source();
		LOG(INFO) << "Opening file " << source;

		std::ifstream infile(source.c_str());
		string line;
		size_t pos1, pos2;
		int file_num = 0;
		int iters_per_epoch = 0;
		while (std::getline(infile, line)) {
			LOG(INFO) << "Loading file " << file_num << "...";
			pos1 = line.find_first_of(' ');
			pos2 = line.find_last_of(' ');
			string image_file_name = line.substr(0, pos1);
			string label_file_name = line.substr(pos1 + 1, pos2 - pos1 - 1);
			string info_file_name = line.substr(pos2 + 1);
			std::ifstream infile_info(root_folder + info_file_name);
			std::vector<int> contour_labels;
			std::vector<int> roi_x0(contour_num_, 0);
			std::vector<int> roi_x1(contour_num_, 0);
			std::vector<int> roi_y0(contour_num_, 0);
			std::vector<int> roi_y1(contour_num_, 0);
			std::vector<int> roi_z0(contour_num_, 0);
			std::vector<int> roi_z1(contour_num_, 0);
			std::vector<bool> organ_exist(contour_num_, false);
			while (std::getline(infile_info, line)) {
				int organ_id = -1;
				pos1 = string::npos;
				for (int i = 0; i < contour_num_; ++i) {
					pos1 = line.find(contour_name_list.name(i));
					if (pos1 != string::npos) {
						organ_id = i;
						organ_exist[organ_id] = true;
						break;
					}
				}
				if (organ_id < 0)
					continue;

				pos1 = line.find_first_of(' ', pos1);
				pos2 = line.find_first_of(' ', pos1 + 1);
				int label_value = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
				contour_labels.push_back(label_value);

				pos1 = pos2;
				pos2 = line.find_first_of(' ', pos1 + 1);
				int x0 = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
				roi_x0[organ_id] = x0;

				pos1 = pos2;
				pos2 = line.find_first_of(' ', pos1 + 1);
				int x1 = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
				roi_x1[organ_id] = x1;

				pos1 = pos2;
				pos2 = line.find_first_of(' ', pos1 + 1);
				int y0 = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
				roi_y0[organ_id] = y0;

				pos1 = pos2;
				pos2 = line.find_first_of(' ', pos1 + 1);
				int y1 = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
				roi_y1[organ_id] = y1;

				pos1 = pos2;
				pos2 = line.find_first_of(' ', pos1 + 1);
				int z0 = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
				roi_z0[organ_id] = z0;

				pos1 = pos2;
				int z1 = atoi(line.substr(pos1 + 1).c_str());
				roi_z1[organ_id] = z1;
			}

			{
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

				resample_spacing[0] = mhd_data_param.resample_spacing_x();
				resample_spacing[1] = mhd_data_param.resample_spacing_y();
				resample_spacing[2] = mhd_data_param.resample_spacing_z();

				if (origin_size[0] * origin_spacing[0] > mhd_data_param.max_width() * mhd_data_param.resample_spacing_x())
					resample_size[0] = mhd_data_param.max_width() * mhd_data_param.resample_spacing_x() / resample_spacing[0];
				else
					resample_size[0] = origin_size[0] * origin_spacing[0] / resample_spacing[0];

				if (origin_size[1] * origin_spacing[1] > mhd_data_param.max_height() * mhd_data_param.resample_spacing_y())
					resample_size[1] = mhd_data_param.max_height() * mhd_data_param.resample_spacing_y() / resample_spacing[1];
				else
					resample_size[1] = origin_size[1] * origin_spacing[1] / resample_spacing[1];

				if (origin_size[2] * origin_spacing[2] > mhd_data_param.max_length() * mhd_data_param.resample_spacing_z())
					resample_size[2] = mhd_data_param.max_length() * mhd_data_param.resample_spacing_z() / resample_spacing[2];
				else
					resample_size[2] = origin_size[2] * origin_spacing[2] / resample_spacing[2];

				resample_origin[0] = origin_origin[0] + origin_size[0] * origin_spacing[0] * 0.5 - resample_size[0] * resample_spacing[0] * 0.5;
				resample_origin[1] = origin_origin[1] + origin_size[1] * origin_spacing[1] * 0.5 - resample_size[1] * resample_spacing[1] * 0.5;
				resample_origin[2] = origin_origin[2] + origin_size[2] * origin_spacing[2] * 0.5 - resample_size[2] * resample_spacing[2] * 0.5;

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

				for (int direct = 0; direct < 3; ++direct)
				{
					VolumeRecord* volume_record = new VolumeRecord;
					volume_record->file_num_ = file_num;
					volume_record->direct_ = direct;
					volume_record->data_ = image;
					volume_record->organ_exist_.assign(organ_exist.begin(), organ_exist.end());
					switch (direct)
					{
					case 0: // axial
						volume_record->bbox_start_pos_.swap(roi_z0);
						volume_record->bbox_end_pos_.swap(roi_z1);

						volume_record->slice_num_ = resample_size[2];
						volume_record->slice_size_[0] = resample_size[0];
						volume_record->slice_size_[1] = resample_size[1];
						volume_record->slice_spacing_[0] = resample_spacing[0];
						volume_record->slice_spacing_[1] = resample_spacing[1];
						volume_record->resample_spacing_ = resample_spacing[2];
						volume_record->resample_origin_ = resample_origin[2];
						volume_record->origin_spacing_ = origin_spacing[2];
						volume_record->origin_origin_ = origin_origin[2];
						break;
					case 1: // coronal
						volume_record->bbox_start_pos_.swap(roi_y0);
						volume_record->bbox_end_pos_.swap(roi_y1);

						volume_record->slice_num_ = resample_size[1];
						volume_record->slice_size_[0] = resample_size[0];
						volume_record->slice_size_[1] = resample_size[2];
						volume_record->slice_spacing_[0] = resample_spacing[0];
						volume_record->slice_spacing_[1] = resample_spacing[2];
						volume_record->resample_spacing_ = resample_spacing[1];
						volume_record->resample_origin_ = resample_origin[1];
						volume_record->origin_spacing_ = origin_spacing[1];
						volume_record->origin_origin_ = origin_origin[1];
						break;
					case 2: // sagittal
						volume_record->bbox_start_pos_.swap(roi_x0);
						volume_record->bbox_end_pos_.swap(roi_x1);

						volume_record->slice_num_ = resample_size[0];
						volume_record->slice_size_[0] = resample_size[1];
						volume_record->slice_size_[1] = resample_size[2];
						volume_record->slice_spacing_[0] = resample_spacing[1];
						volume_record->slice_spacing_[1] = resample_spacing[2];
						volume_record->resample_spacing_ = resample_spacing[0];
						volume_record->resample_origin_ = resample_origin[0];
						volume_record->origin_spacing_ = origin_spacing[0];
						volume_record->origin_origin_ = origin_origin[0];
						break;
					}
					volume_list_.push_back(volume_record);
					volume_id_list_.push_back(volume_list_.size() - 1);

					if (volume_record->slice_num_ % batch_size == 0)
						iters_per_epoch += int(volume_record->slice_num_ / batch_size);
					else
						iters_per_epoch += int(volume_record->slice_num_ / batch_size) + 1;
				}
			}
			file_num++;
		}

		CHECK(!volume_list_.empty()) << "File is empty";

		if (this->phase_ == TRAIN && mhd_data_param.shuffle()) {
			// randomly shuffle data
			LOG(INFO) << "Shuffling data";
			const unsigned int prefetch_rng_seed = caffe_rng_rand();
			prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
			ShuffleImages();
		}

		LOG(INFO) << "A total of " << volume_list_.size() << " images, " << iters_per_epoch << " iterations per epoch.";

		volume_id_ = 0;
		start_slice_id_ = 0;

		vector<int> data_shape(4);
		data_shape[0] = 1;
		data_shape[1] = 1;
		data_shape[2] = 500;
		data_shape[3] = 500;
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
			label_shape[2] = 2;
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
	void MHDSliceDataLayer<Dtype>::ShuffleImages() {
		caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		shuffle(volume_id_list_.begin(), volume_id_list_.end(), prefetch_rng);
	}

	// This function is called on prefetch thread
	template <typename Dtype>
	void MHDSliceDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

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

		const int volume_num = volume_list_.size();
		CHECK_GT(volume_num, volume_id_);
		VolumeRecord* volume_record = volume_list_[volume_id_list_[volume_id_]];
		ImageType::Pointer image = volume_record->data_;
		VoxelType* image_buffer = image->GetBufferPointer();
		ImageType::SizeType volume_size = image->GetBufferedRegion().GetSize();
		const MHDDataParameter& mhd_data_param = this->layer_param_.mhd_data_param();
		const int batch_size = min((int)mhd_data_param.batch_size(), volume_record->slice_num_ - start_slice_id_);
		const string& root_folder = mhd_data_param.root_folder();

		ImageSliceType::SizeType slice_size;
		ImageSliceType::RegionType slice_region;
		slice_size[0] = volume_record->slice_size_[0];
		slice_size[1] = volume_record->slice_size_[1];
		slice_region.SetSize(slice_size);
		ImageSliceType::SpacingType slice_spacing;
		slice_spacing[0] = volume_record->slice_spacing_[0];
		slice_spacing[1] = volume_record->slice_spacing_[1];

		vector<int> data_shape(4);
		data_shape[0] = batch_size;
		data_shape[1] = 1;
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
				label_shape[0] = 1;
				label_shape[1] = 1;
				label_shape[2] = 2;
			}
			batch->label_.Reshape(label_shape);
			label_data = batch->label_.mutable_cpu_data();
			memset(label_data, 0, sizeof(Dtype) * batch->label_.count());
		}

		for (int batch_id = 0; batch_id < batch_size; ++batch_id)
		{
			int i = start_slice_id_ + batch_id;

			ImageSliceType::Pointer slice = ImageSliceType::New();
			slice->SetRegions(slice_region);
			slice->SetSpacing(slice_spacing);
			slice->Allocate();
			VoxelType* slice_buffer = slice->GetBufferPointer();

			// extract slices
			switch (volume_record->direct_)
			{
			case 0: // axial
			{
				for (int j = 0; j < slice_size[1]; ++j)
				{
					for (int k = 0; k < slice_size[0]; ++k)
					{
						slice_buffer[j * slice_size[0] + k] = image_buffer[i * volume_size[1] * volume_size[0] + j * volume_size[0] + k];
					}
				}
				break;
			}
			case 1: // coronal
			{
				for (int j = 0; j < slice_size[1]; ++j)
				{
					for (int k = 0; k < slice_size[0]; ++k)
					{
						slice_buffer[j * slice_size[0] + k] = image_buffer[j * volume_size[1] * volume_size[0] + i * volume_size[0] + k];
					}
				}
				break;
			}
			case 2: // sagittal
			{
				for (int j = 0; j < slice_size[1]; ++j)
				{
					for (int k = 0; k < slice_size[0]; ++k)
					{
						slice_buffer[j * slice_size[0] + k] = image_buffer[j * volume_size[1] * volume_size[0] + k * volume_size[0] + i];
					}
				}
				break;
			}
			}

			std::vector<int> labels;
			labels.resize(contour_num_, -1);
			int ii = (volume_record->resample_origin_ + i * volume_record->resample_spacing_ - volume_record->origin_origin_) / volume_record->origin_spacing_;
			for (int j = 0; j < contour_num_; ++j)
			{
				if (volume_record->organ_exist_[j])
				{
					if (ii >= volume_record->bbox_start_pos_[j] && ii <= volume_record->bbox_end_pos_[j])
					{
						labels[j] = 1;
					}
					else
					{
						labels[j] = 0;
					}
				}
			}

			// augmentation
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

			slice_buffer = resample_slice->GetBufferPointer();
			int batch_data_start_ind = batch_id * slice_size[1] * slice_size[0];
			for (int j = 0; j < slice_size[1] * slice_size[0]; ++j)
			{
				batch_data[batch_data_start_ind + j] = (Dtype)(slice_buffer[j]);
			}

			if (output_labels_)
			{
				if (this->phase_ == TRAIN)
				{
					for (int j = 0; j < contour_num_; ++j)
					{
						label_data[batch_id * contour_num_ + j] = labels[j];
					}
				}
				else
				{
					label_data[0] = volume_record->file_num_;
					label_data[1] = volume_record->direct_;
				}
			}
		}

		start_slice_id_ += batch_size;
		if (start_slice_id_ >= volume_record->slice_num_) {
			start_slice_id_ = 0;
			volume_id_ += 1;
			if (volume_id_ >= volume_num)
			{
				volume_id_ = 0;
				if (this->phase_ == TRAIN && mhd_data_param.shuffle()) {
					ShuffleImages();
				}
			}
		}
	}

	INSTANTIATE_CLASS(MHDSliceDataLayer);
	REGISTER_LAYER_CLASS(MHDSliceData);

}  // namespace caffe
