#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/multi_slice_predict_layer.hpp"
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
void MultiSlicePredictLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	itk::ObjectFactoryBase::RegisterFactory(itk::MetaImageIOFactory::New());
	typedef itk::ImageFileReader<ImageType> ImageReaderType;

	const SlicePredictParameter& slice_predict_param = this->layer_param_.slice_predict_param();
	std::ifstream infile(slice_predict_param.source().c_str());
	string line;
	size_t pos1;
	while (std::getline(infile, line)) {
		pos1 = line.find_first_of(' ');
		string image_file_name = line.substr(0, pos1);
		file_names_.push_back(image_file_name);
	}
	line_id_ = 0;
}

template <typename Dtype>
void MultiSlicePredictLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
  // Reshaping happens during the call to forward.
}

template <typename Dtype>
void MultiSlicePredictLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const SlicePredictParameter& slice_predict_param = this->layer_param_.slice_predict_param();
	const Dtype* info_data = bottom[0]->cpu_data();
	const ContourNameList& contour_name_list = slice_predict_param.contour_name_list();
	const int class_num = contour_name_list.name_size();
	const int batch_size = bottom[1]->shape(0);
	const Dtype* pd_a_data = bottom[1]->cpu_data();
	const Dtype* pd_c_data = bottom[2]->cpu_data();
	const Dtype* pd_s_data = bottom[3]->cpu_data();

	for (int batch_ind = 0; batch_ind < batch_size; ++batch_ind)
	{
		const string& file_name = file_names_[line_id_];
		string abs_file_name = slice_predict_param.output_path() + file_name + ".pred.txt";
#ifdef OUTPUT_PROJECTION
		string axial_file_name = slice_predict_param.output_path() + file_name + ".a.txt";
		string coronal_file_name = slice_predict_param.output_path() + file_name + ".c.txt";
		string sagittal_file_name = slice_predict_param.output_path() + file_name + ".s.txt";
		ofstream axial_file(axial_file_name, ios::out | ios::trunc);
		ofstream coronal_file(coronal_file_name, ios::out | ios::trunc);
		ofstream sagittal_file(sagittal_file_name, ios::out | ios::trunc);
		axial_file.is_open();
		coronal_file.is_open();
		sagittal_file.is_open();
#endif
		ofstream file(abs_file_name, ios::out | ios::trunc);
		if (file.is_open())
		{

			int size[3];
			size[0] = info_data[2];
			size[1] = info_data[1];
			size[2] = info_data[0];
			CHECK_EQ(size[2] * class_num, bottom[1]->shape(2));
			CHECK_EQ(size[1] * class_num, bottom[2]->shape(2));
			CHECK_EQ(size[0] * class_num, bottom[3]->shape(2));
			double spacing[3];
			spacing[0] = info_data[5];
			spacing[1] = info_data[4];
			spacing[2] = info_data[3];
			double origin[3];
			origin[0] = info_data[8];
			origin[1] = info_data[7];
			origin[2] = info_data[6];
			int origin_size[3];
			origin_size[0] = info_data[11];
			origin_size[1] = info_data[10];
			origin_size[2] = info_data[9];
			double origin_spacing[3];
			origin_spacing[0] = info_data[14];
			origin_spacing[1] = info_data[13];
			origin_spacing[2] = info_data[12];
			double origin_origin[3];
			origin_origin[0] = info_data[17];
			origin_origin[1] = info_data[16];
			origin_origin[2] = info_data[15];

			for (int cls = 0; cls < class_num; ++cls)
			{
				int x0 = -1;
				int x1 = -1;
				int y0 = -1;
				int y1 = -1;
				int z0 = -1;
				int z1 = -1;
				std::vector<int> start_z_pos;
				std::vector<int> end_z_pos;
				for (int pos = 0; pos < size[2]; ++pos)
				{
					if (pd_a_data[(class_num + cls) * size[2] + pos] > 0.5 && (pos == 0 || pd_a_data[(class_num + cls) * size[2] + pos - 1] <= 0.5))
					{
						start_z_pos.push_back(pos);
					}
					if (pd_a_data[(class_num + cls) * size[2] + pos] > 0.5 && (pos == size[2] - 1 || pd_a_data[(class_num + cls) * size[2] + pos + 1] <= 0.5))
					{
						end_z_pos.push_back(pos);
					}
				}
				int max_z_length = 0;
				for (int i = 0; i < start_z_pos.size(); ++i)
				{
					if (max_z_length < end_z_pos[i] - start_z_pos[i] + 1)
					{
						max_z_length = end_z_pos[i] - start_z_pos[i] + 1;
						z0 = start_z_pos[i];
						z1 = end_z_pos[i];
					}
				}

				std::vector<int> start_y_pos;
				std::vector<int> end_y_pos;
				for (int pos = 0; pos < size[1]; ++pos)
				{
					if (pd_c_data[(class_num + cls) * size[1] + pos] > 0.5 && (pos == 0 || pd_c_data[(class_num + cls) * size[1] + pos - 1] <= 0.5))
					{
						start_y_pos.push_back(pos);
					}
					if (pd_c_data[(class_num + cls) * size[1] + pos] > 0.5 && (pos == size[1] - 1 || pd_c_data[(class_num + cls) * size[1] + pos + 1] <= 0.5))
					{
						end_y_pos.push_back(pos);
					}
				}
				int max_y_length = 0;
				for (int i = 0; i < start_y_pos.size(); ++i)
				{
					if (max_y_length < end_y_pos[i] - start_y_pos[i] + 1)
					{
						max_y_length = end_y_pos[i] - start_y_pos[i] + 1;
						y0 = start_y_pos[i];
						y1 = end_y_pos[i];
					}
				}

				std::vector<int> start_x_pos;
				std::vector<int> end_x_pos;
				for (int pos = 0; pos < size[0]; ++pos)
				{
					if (pd_s_data[(class_num + cls) * size[0] + pos] > 0.5 && (pos == 0 || pd_s_data[(class_num + cls) * size[0] + pos - 1] <= 0.5))
					{
						start_x_pos.push_back(pos);
					}
					if (pd_s_data[(class_num + cls) * size[0] + pos] > 0.5 && (pos == size[0] - 1 || pd_s_data[(class_num + cls) * size[0] + pos + 1] <= 0.5))
					{
						end_x_pos.push_back(pos);
					}
				}
				int max_x_length = 0;
				for (int i = 0; i < start_x_pos.size(); ++i)
				{
					if (max_x_length < end_x_pos[i] - start_x_pos[i] + 1)
					{
						max_x_length = end_x_pos[i] - start_x_pos[i] + 1;
						x0 = start_x_pos[i];
						x1 = end_x_pos[i];
					}
				}
				if (x0 < 0 || x1 < 0 || y0 < 0 || y1 < 0 || z0 < 0 || z1 < 0)
				{
					continue;
				}

				double prob = 0;
				int prob_num = 0;
				for (int i = x0; i <= x1; ++i)
				{
					prob_num += 1;
					prob += pd_s_data[(class_num + cls) * size[0] + i];
				}
				for (int i = y0; i <= y1; ++i)
				{
					prob_num += 1;
					prob += pd_c_data[(class_num + cls) * size[1] + i];
				}
				for (int i = z0; i <= z1; ++i)
				{
					prob_num += 1;
					prob += pd_a_data[(class_num + cls) * size[2] + i];
				}
				prob = prob / prob_num;

				file << cls + 1 << " ";
				file << (x0 * spacing[0] + origin[0] - origin_origin[0]) / origin_spacing[0] << " ";
				file << (y0 * spacing[1] + origin[1] - origin_origin[1]) / origin_spacing[1] << " ";
				file << (z0 * spacing[2] + origin[2] - origin_origin[2]) / origin_spacing[2] << " ";
				file << (x1 * spacing[0] + origin[0] - origin_origin[0]) / origin_spacing[0] << " ";
				file << (y1 * spacing[1] + origin[1] - origin_origin[1]) / origin_spacing[1] << " ";
				file << (z1 * spacing[2] + origin[2] - origin_origin[2]) / origin_spacing[2] << " ";
				file << prob << std::endl;

#ifdef OUTPUT_PROJECTION
				axial_file << cls + 1;
				for (int pos = 0; pos < size[2]; ++pos)
				{
					axial_file << " " << pd_a_data[(class_num + cls) * size[2] + pos];
				}
				axial_file << std::endl;
				coronal_file << cls + 1;
				for (int pos = 0; pos < size[1]; ++pos)
				{
					coronal_file << " " << pd_c_data[(class_num + cls) * size[1] + pos];
				}
				coronal_file << std::endl;
				sagittal_file << cls + 1;
				for (int pos = 0; pos < size[0]; ++pos)
				{
					sagittal_file << " " << pd_s_data[(class_num + cls) * size[0] + pos];
				}
				sagittal_file << std::endl;
#endif
			}
			file.close();
		}
#ifdef OUTPUT_PROJECTION
		axial_file.close();
		coronal_file.close();
		sagittal_file.close();
#endif

		info_data += bottom[0]->shape(1);
		pd_a_data += 2 * bottom[1]->shape(2);
		pd_c_data += 2 * bottom[2]->shape(2);
		pd_s_data += 2 * bottom[3]->shape(2);
		line_id_ += 1;
	}
}

template <typename Dtype>
void MultiSlicePredictLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(MultiSlicePredictLayer);
REGISTER_LAYER_CLASS(MultiSlicePredict);

}  // namespace caffe