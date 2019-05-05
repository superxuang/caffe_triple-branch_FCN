#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/roi_align_layer.hpp"


using std::max;
using std::min;

namespace caffe {

	template <typename Dtype>
	__global__ void ROIAlignForward(const int nthreads, const Dtype* bottom_data,
		const Dtype spatial_scale_xy, const Dtype spatial_scale_z, const int channels, 
		const int length, const int height, const int width, 
		const int pooled_length, const int pooled_height, const int pooled_width,
		const Dtype* bottom_rois, const int interpolate_times, Dtype* top_data, int* argmax_data, Dtype* w_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {

			// (n, c, pl, ph, pw) is an element in the pooled output
			int pw = index % pooled_width;
			int ph = (index / pooled_width) % pooled_height;
			int pl = (index / pooled_width / pooled_height) % pooled_length;
			int c = (index / pooled_width / pooled_height / pooled_length) % channels;
			int n = index / pooled_width / pooled_height / pooled_length / channels;

			bottom_rois += n * 7;
			int roi_batch_ind = bottom_rois[0];

			Dtype roi_start_w = bottom_rois[1] * spatial_scale_xy;
			Dtype roi_start_h = bottom_rois[2] * spatial_scale_xy;
			Dtype roi_start_l = bottom_rois[3] * spatial_scale_z;
			Dtype roi_end_w = bottom_rois[4] * spatial_scale_xy;
			Dtype roi_end_h = bottom_rois[5] * spatial_scale_xy;
			Dtype roi_end_l = bottom_rois[6] * spatial_scale_z;
			// clipping
			roi_start_w = max(roi_start_w, Dtype(0)); 
			roi_start_h = max(roi_start_h, Dtype(0));
			roi_start_l = max(roi_start_l, Dtype(0));
			int img_width = round(width / spatial_scale_xy);
			int img_height = round(height / spatial_scale_xy);
			int img_length = round(length / spatial_scale_z);
			roi_end_w = min(Dtype(img_width - 1), roi_end_w);
			roi_end_h = min(Dtype(img_height - 1), roi_end_h);
			roi_end_l = min(Dtype(img_length - 1), roi_end_l);

			Dtype roi_length = max(roi_end_l - roi_start_l + 1, Dtype(1));
			Dtype roi_height = max(roi_end_h - roi_start_h + 1, Dtype(1));
			Dtype roi_width = max(roi_end_w - roi_start_w + 1, Dtype(1));
			const Dtype bin_size_l = static_cast<Dtype>(roi_length)
				/ static_cast<Dtype>(roi_length);
			const Dtype bin_size_h = static_cast<Dtype>(roi_height)
				/ static_cast<Dtype>(pooled_height);
			const Dtype bin_size_w = static_cast<Dtype>(roi_width)
				/ static_cast<Dtype>(pooled_width);

			bottom_data += (roi_batch_ind * channels + c) * length * height * width;

			double argmax_temp_data[8];
			double w_temp_data[8];
			double start_x = 0.25, start_y = 0.25, start_z = 0.25;
			if (interpolate_times == 1) {
				start_x = 0.5;
				start_y = 0.5;
				start_z = 0.5;
			}
			Dtype dfValue = 0, maxValue = 0;
			for (int inter_index = 0; inter_index < interpolate_times; ++inter_index) {
				int index_x = inter_index / 4;
				int index_y = inter_index / 2;
				int index_z = inter_index % 2;
				Dtype off_x = index_x * 0.5 + start_x;
				Dtype off_y = index_y * 0.5 + start_y;
				Dtype off_z = index_z * 0.5 + start_z;
				Dtype lcenter = static_cast<Dtype>(pl + off_x)* bin_size_l;
				Dtype hcenter = static_cast<Dtype>(ph + off_y)* bin_size_h;
				Dtype wcenter = static_cast<Dtype>(pw + off_z)* bin_size_w;

				lcenter = min(max(lcenter + roi_start_l, Dtype(0)), Dtype(length - 1));
				hcenter = min(max(hcenter + roi_start_h, Dtype(0)), Dtype(height - 1));
				wcenter = min(max(wcenter + roi_start_w, Dtype(0)), Dtype(width - 1));

				int lstart = min(max(lcenter, Dtype(0)), Dtype(length - 1));
				int hstart = min(max(hcenter, Dtype(0)), Dtype(height - 1));
				int wstart = min(max(wcenter, Dtype(0)), Dtype(width - 1));
				int lend = min(max(lstart + 1, 0), length - 1);
				int hend = min(max(hstart + 1, 0), height - 1);
				int wend = min(max(wstart + 1, 0), width - 1);

				Dtype fX0 = wcenter - wstart;
				Dtype fX1 = wend - wcenter;
				Dtype fY0 = hcenter - hstart;
				Dtype fY1 = hend - hcenter;
				Dtype fZ0 = lcenter - lstart;
				Dtype fZ1 = lend - lcenter;
				Dtype fFactorA = fZ1 * fY1 * fX1;
				Dtype fFactorB = fZ1 * fY1 * fX0;
				Dtype fFactorC = fZ1 * fY0 * fX1;
				Dtype fFactorD = fZ1 * fY0 * fX0;
				Dtype fFactorE = fZ0 * fY1 * fX1;
				Dtype fFactorF = fZ0 * fY1 * fX0;
				Dtype fFactorG = fZ0 * fY0 * fX1;
				Dtype fFactorH = fZ0 * fY0 * fX0;

				dfValue = bottom_data[lstart * height * width + hstart * width + wstart] * fFactorA
					+ bottom_data[lstart * height * width + hstart * width + wend] * fFactorB
					+ bottom_data[lstart * height * width + hend * width + wstart] * fFactorC
					+ bottom_data[lstart * height * width + hend * width + wend] * fFactorD
					+ bottom_data[lend * height * width + hstart * width + wstart] * fFactorE
					+ bottom_data[lend * height * width + hstart * width + wend] * fFactorF
					+ bottom_data[lend * height * width + hend * width + wstart] * fFactorG
					+ bottom_data[lend * height * width + hend * width + wend] * fFactorH;

				if (inter_index == 0) {
					maxValue = dfValue - 1;
				}

				argmax_temp_data[0] = lstart * height * width + hstart * width + wstart;
				argmax_temp_data[1] = lstart * height * width + hstart * width + wend;
				argmax_temp_data[2] = lstart * height * width + hend * width + wstart;
				argmax_temp_data[3] = lstart * height * width + hend * width + wend;
				argmax_temp_data[4] = lend * height * width + hstart * width + wstart;
				argmax_temp_data[5] = lend * height * width + hstart * width + wend;
				argmax_temp_data[6] = lend * height * width + hend * width + wstart;
				argmax_temp_data[7] = lend * height * width + hend * width + wend;

				w_temp_data[0] = fFactorA;
				w_temp_data[1] = fFactorB;
				w_temp_data[2] = fFactorC;
				w_temp_data[3] = fFactorD;
				w_temp_data[4] = fFactorE;
				w_temp_data[5] = fFactorF;
				w_temp_data[6] = fFactorG;
				w_temp_data[7] = fFactorH;

				if (dfValue > maxValue || inter_index == 0) {
					maxValue = dfValue;
					top_data[index] = dfValue;
					for (int s = 0; s < 8; ++s) {
						w_data[8 * index + s] = w_temp_data[s];
						argmax_data[8 * index + s] = argmax_temp_data[s];
					}
				}
			}
		}
	}

	template <typename Dtype>
	__global__ void ROICubicForward(const int nthreads, const Dtype* bottom_data,
		const Dtype spatial_scale_xy, const Dtype spatial_scale_z, const int channels, 
		const int length, const int height, const int width,
		const int pooled_length, const int pooled_height, const int pooled_width,
		const Dtype* bottom_rois, const int interpolate_times, Dtype* top_data, int* argmax_data, Dtype* w_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {

			// (n, c, pl, ph, pw) is an element in the pooled output
			int pw = index % pooled_width;
			int ph = (index / pooled_width) % pooled_height;
			int pl = (index / pooled_width / pooled_height) % pooled_length;
			int c = (index / pooled_width / pooled_height / pooled_length) % channels;
			int n = index / pooled_width / pooled_height / pooled_length / channels;

			bottom_rois += n * 7;
			int roi_batch_ind = bottom_rois[0];

			Dtype roi_start_w = bottom_rois[1] * spatial_scale_xy;
			Dtype roi_start_h = bottom_rois[2] * spatial_scale_xy;
			Dtype roi_start_l = bottom_rois[3] * spatial_scale_z;
			Dtype roi_end_w = bottom_rois[4] * spatial_scale_xy;
			Dtype roi_end_h = bottom_rois[5] * spatial_scale_xy;
			Dtype roi_end_l = bottom_rois[6] * spatial_scale_z;
			// clipping
			roi_start_w = max(roi_start_w, Dtype(0));
			roi_start_h = max(roi_start_h, Dtype(0));
			roi_start_l = max(roi_start_l, Dtype(0));
			int img_width = round(width / spatial_scale_xy);
			int img_height = round(height / spatial_scale_xy);
			int img_length = round(length / spatial_scale_z);
			roi_end_w = min(Dtype(img_width - 1), roi_end_w);
			roi_end_h = min(Dtype(img_height - 1), roi_end_h);
			roi_end_l = min(Dtype(img_length - 1), roi_end_l);

			Dtype roi_length = max(roi_end_l - roi_start_l + 1, Dtype(1));
			Dtype roi_height = max(roi_end_h - roi_start_h + 1, Dtype(1));
			Dtype roi_width = max(roi_end_w - roi_start_w + 1, Dtype(1));
			const Dtype bin_size_l = static_cast<Dtype>(roi_length)
				/ static_cast<Dtype>(roi_length);
			const Dtype bin_size_h = static_cast<Dtype>(roi_height)
				/ static_cast<Dtype>(pooled_height);
			const Dtype bin_size_w = static_cast<Dtype>(roi_width)
				/ static_cast<Dtype>(pooled_width);

			bottom_data += (roi_batch_ind * channels + c) * length * height * width;
			double argmax_temp_data[64];
			double w_temp_data[64];
			double start_x = 0.25, start_y = 0.25, start_z = 0.25;
			if (interpolate_times == 1) {
				start_x = 0.5;
				start_y = 0.5;
				start_z = 0.5;
			}
			Dtype dfCubicValue = 0, maxValue = 0;
			for (int inter_index = 0; inter_index < interpolate_times; ++inter_index) {
				int index_x = inter_index / 4;
				int index_y = inter_index / 2;
				int index_z = inter_index % 2;
				Dtype off_x = index_x * 0.5 + start_x;
				Dtype off_y = index_y * 0.5 + start_y;
				Dtype off_z = index_z * 0.5 + start_z;
				Dtype lcenter = static_cast<Dtype>(pl + off_x)* bin_size_l;
				Dtype hcenter = static_cast<Dtype>(ph + off_y)* bin_size_h;
				Dtype wcenter = static_cast<Dtype>(pw + off_z)* bin_size_w;

				lcenter = min(max(lcenter + roi_start_l, Dtype(0)), Dtype(length - 1));
				hcenter = min(max(hcenter + roi_start_h, Dtype(0)), Dtype(height - 1));
				wcenter = min(max(wcenter + roi_start_w, Dtype(0)), Dtype(width - 1));

				int i = wcenter;
				int j = hcenter;
				int k = lcenter;

				/*get adjacent 64 values*/
				double values[4][4][4];
				int temp_c, temp_r, temp_rr;
				for (int rr = k - 1, ss = 0; rr <= k + 2; rr++, ss++){
					for (int r = j - 1, s = 0; r <= j + 2; r++, s++){
						for (int c = i - 1, t = 0; c <= i + 2; c++, t++){
							//todo: ??64?,????
							temp_c = min(max(Dtype(c), Dtype(0)), Dtype(width - 1));
							temp_r = min(max(Dtype(r), Dtype(0)), Dtype(height - 1));
							temp_rr = min(max(Dtype(rr), Dtype(0)), Dtype(length - 1));
							values[ss][s][t] = bottom_data[temp_rr*height*width + temp_r*width + temp_c];
							argmax_temp_data[ss * 16 + s * 4 + t] = temp_rr*height*width + temp_r*width + temp_c;
						}
					}
				}

				/*calc the coeff*/
				double u = wcenter - i;
				double v = hcenter - j;
				double vv = lcenter - k;
				double A[4], C[4], CC[4];
				for (int distance = 1, s = 0; distance >= -2; distance--, s++){
					A[s] = cubic_coeff_gpu(u + distance);
					C[s] = cubic_coeff_gpu(v + distance);
					CC[s] = cubic_coeff_gpu(vv + distance);
				}

				dfCubicValue = 0;
				for (int ss = 0; ss < 4; ss++) {
					for (int s = 0; s < 4; s++) {
						for (int t = 0; t < 4; t++) {
							dfCubicValue += values[ss][s][t] * A[t] * C[s] * CC[ss];
							w_temp_data[ss * 16 + s * 4 + t] = A[t] * C[s] * CC[ss];
						}
					}
				}
				if (dfCubicValue > maxValue || inter_index == 0) {
					maxValue = dfCubicValue;
					top_data[index] = dfCubicValue;
					for (int s = 0; s < 64; ++s) {
						w_data[64 * index + s] = w_temp_data[s];
						argmax_data[64 * index + s] = argmax_temp_data[s];
					}
				}
			}
		}
	}

	template <typename Dtype>
	void ROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* bottom_rois = bottom[1]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		int* argmax_data = bili_idx.mutable_gpu_data();
		Dtype* w_data = bili_w.mutable_gpu_data();
		int count = top[0]->count();
		int interpolate_times = is_multi_interpolate ? 8 : 1;
		// NOLINT_NEXT_LINE(whitespace/operators)
		if (bi_type == BiCubic) {
			ROICubicForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, bottom_data, spatial_scale_xy_, spatial_scale_z_, channels_, length_, height_, width_,
				pooled_length_, pooled_height_, pooled_width_, bottom_rois, interpolate_times, top_data, argmax_data, w_data);
		}
		else {
			ROIAlignForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, bottom_data, spatial_scale_xy_, spatial_scale_z_, channels_, length_, height_, width_,
				pooled_length_, pooled_height_, pooled_width_, bottom_rois, interpolate_times, top_data, argmax_data, w_data);
		}
		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	__global__ void ROIAlignBackward(const int nthreads, const Dtype* top_diff,
		const int* argmax_data, const Dtype* w_data, const int num_rois, 
		const int channels, 
		const int length, const int height, const int width,
		const int pooled_length, const int pooled_height, const int pooled_width,
		const int w_num, Dtype* bottom_diff, const Dtype* bottom_rois) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			// (n, c, pl, ph, pw) is an element in the pooled output
			//int pw = index % pooled_width;
			//int ph = (index / pooled_width) % pooled_height;
			//int pl = (index / pooled_width / pooled_height) % pooled_length;
			int c = (index / pooled_width / pooled_height / pooled_length) % channels;
			int n = index / pooled_width / pooled_height / pooled_length / channels;

			bottom_rois += n * 7;
			int roi_batch_ind = bottom_rois[0];

			for (int i = 0; i < w_num; ++i) {
				if (argmax_data[w_num * index + i] >= 0) {
					int offset_bottom = (roi_batch_ind * channels + c) * length * height
						* width + argmax_data[w_num * index + i];
					bottom_diff[offset_bottom] += top_diff[index] * w_data[w_num * index + i];
				}
			}
		}
	}

	template <typename Dtype>
	void ROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0]) {
			return;
		}
		const Dtype* bottom_rois = bottom[1]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const int count = bottom[0]->count();
		caffe_gpu_set(count, Dtype(0.), bottom_diff);
		const int* argmax_data = bili_idx.gpu_data();
		const Dtype* w_data = bili_w.gpu_data();
		const int top_count = top[0]->count();
		int w_num = 8;
		if (bi_type == BiCubic) {
			w_num = 64;
		}
		// NOLINT_NEXT_LINE(whitespace/operators)
		ROIAlignBackward<Dtype> << <CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS >> >(
			top_count, top_diff, argmax_data, w_data, top[0]->num(), channels_,
			length_, height_, width_, pooled_length_, pooled_height_, pooled_width_, w_num, bottom_diff, bottom_rois);
		CUDA_POST_KERNEL_CHECK;
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignLayer);

}  // namespace caffe