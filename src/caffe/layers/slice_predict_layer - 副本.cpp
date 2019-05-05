#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/slice_predict_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <itkImage.h>
#include <itkMetaImageIOFactory.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

using std::sqrt;
using std::floor;
using std::ceil;
using std::min;
using std::max;
using std::exp;
using std::log;

namespace caffe {

template <typename Dtype>
void SlicePredictLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	itk::ObjectFactoryBase::RegisterFactory(itk::MetaImageIOFactory::New());
	typedef itk::ImageFileReader<ImageType> ImageReaderType;
	typedef itk::ImageFileReader<LabelType> LabelReaderType;

	const SlicePredictParameter& slice_predict_param = this->layer_param_.slice_predict_param();
	const unsigned int rng_seed = caffe_rng_rand();
	rng_.reset(new Caffe::RNG(rng_seed));
	const int contour_num_ = slice_predict_param.contour_name_list().name_size();
	std::ifstream infile(slice_predict_param.source().c_str());
	string line;
	size_t pos1, pos2;
	while (std::getline(infile, line)) {
		pos1 = line.find_first_of(' ');
		pos2 = line.find_last_of(' ');
		string image_file_name = line.substr(0, pos1);
		string label_file_name = line.substr(pos1 + 1, pos2 - pos1 - 1);
		string info_file_name = line.substr(pos2 + 1);
		std::ifstream infile_info(slice_predict_param.input_path() + info_file_name);
		std::vector<int> contour_labels;
		std::vector<int> exist_contours;
		while (std::getline(infile_info, line)) {
			pos1 = string::npos;
			for (int i = 0; i < contour_num_; ++i) {
				pos1 = line.find(slice_predict_param.contour_name_list().name(i));
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
		}
		if (!contour_labels.empty()) {
			ImageReaderType::Pointer reader_image = ImageReaderType::New();
			reader_image->SetFileName(slice_predict_param.input_path() + image_file_name);
			reader_image->UpdateOutputInformation();
			ImageType::Pointer image = reader_image->GetOutput();
			const ImageType::SizeType& image_size = image->GetLargestPossibleRegion().GetSize();
			const ImageType::SpacingType& image_spacing = image->GetSpacing();

			size_t pos = image_file_name.find_first_of('/');
			image_file_name = image_file_name.substr(pos + 1);
			for (int i = 0; i < 3; ++i)
			{
				for (int j = 0; j < image_size[2 - i]; ++j)
				{
					lines_.push_back(std::make_pair(image_file_name, std::make_pair(i, j)));
				}
			}
		}
	}
	lines_id_ = 0;
	output_buffer_.clear();
}

template <typename Dtype>
void SlicePredictLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
  // Reshaping happens during the call to forward.
}

template <typename Dtype>
void SlicePredictLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const SlicePredictParameter& slice_predict_param = this->layer_param_.slice_predict_param();
	const int batch_size = bottom[0]->shape(0);
	CHECK_EQ(bottom[0]->shape(1), 2);
	const int class_num = bottom[0]->shape(2);
	const Dtype* batch_data = bottom[0]->cpu_data();
	Dtype* label_data = NULL;
	bool output_label = bottom.size() > 1;
	if (output_label)
	{
		label_data = bottom[1]->mutable_cpu_data();
	}
	const int line_size = lines_.size();
	for (int batch_ind = 0; batch_ind < batch_size; ++batch_ind)
	{
		if (lines_id_ >= line_size)
		{
			break;
		}
		const string& file_name = lines_[lines_id_].first;
		const int& direction = lines_[lines_id_].second.first;
		const int& slice_ind = lines_[lines_id_].second.second;

		string abs_file_name = slice_predict_param.output_path() + file_name + ".slice.txt";
		switch (direction)
		{
		case 0: // axial
			abs_file_name = slice_predict_param.output_path() + file_name + ".axial.txt";
			break;
		case 1: // coronal
			abs_file_name = slice_predict_param.output_path() + file_name + ".coronal.txt";
			break;
		case 2: // sagittal
			abs_file_name = slice_predict_param.output_path() + file_name + ".sagittal.txt";
			break;
		}

		for (int i = 0; i < class_num; ++i)
		{
			output_buffer_ << batch_data[batch_ind * 2 * class_num + class_num * 1 + i] << " ";
		}
		output_buffer_ << std::endl;
		if (lines_id_ >= line_size - 1 || file_name != lines_[lines_id_ + 1].first || direction != lines_[lines_id_ + 1].second.first)
		{
			ofstream file(abs_file_name, ios::out | ios::trunc);
			if (file.is_open())
			{
				file << output_buffer_.str();
				file.close();
			}
			output_buffer_.str("");
			LOG(INFO) << abs_file_name;
		}

		lines_id_ += 1;
	}
}

template <typename Dtype>
void SlicePredictLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(SlicePredictLayer);
REGISTER_LAYER_CLASS(SlicePredict);

}  // namespace caffe