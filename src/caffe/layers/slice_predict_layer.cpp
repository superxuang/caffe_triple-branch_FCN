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

	const SlicePredictParameter& slice_predict_param = this->layer_param_.slice_predict_param();
	std::ifstream infile(slice_predict_param.source().c_str());
	string line;
	size_t pos1;
	while (std::getline(infile, line)) {
		pos1 = line.find_first_of(' ');
		string file_name = line.substr(0, pos1);
		for (int direct = 0; direct < 3; ++direct)
		{
			string abs_file_name = slice_predict_param.output_path() + file_name + ".slice.txt";
			switch (direct)
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

			output_file_names_.push_back(abs_file_name);

			remove(abs_file_name.c_str());
		}
	}
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
	int class_num = bottom[0]->shape(2);

	const Dtype* batch_data = bottom[0]->cpu_data();
	const Dtype* info_data = bottom[1]->cpu_data();
	const int file_num = info_data[0];
	const int direct = info_data[1];
	CHECK_GE(file_num, 0);
	CHECK_LT(file_num, output_file_names_.size());
	const string& abs_file_name = output_file_names_[file_num * 3 + direct];
	for (int batch_ind = 0; batch_ind < batch_size; ++batch_ind)
	{
		ofstream file(abs_file_name, ios::out | ios::app);
		if (file.is_open())
		{
			for (int i = 0; i < class_num; ++i)
			{
				file << batch_data[batch_ind * 2 * class_num + class_num * 1 + i] << " ";
			}
			file << std::endl;
			file.close();
		}
	}
}

template <typename Dtype>
void SlicePredictLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(SlicePredictLayer);
REGISTER_LAYER_CLASS(SlicePredict);

}  // namespace caffe