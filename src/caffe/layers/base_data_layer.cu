#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
		// Reshape to loaded data.
		top[0]->ReshapeLike(batch->data_);
		// Copy the data
		caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
			top[0]->mutable_gpu_data());
		if (this->output_labels_) {
			// Reshape to loaded labels.
			top[1]->ReshapeLike(batch->label_);
			// Copy the labels.
			caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
				top[1]->mutable_gpu_data());
		}
		// Ensure the copy is synchronous wrt the host, so that the next batch isn't
		// copied in meanwhile.
		CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
		prefetch_free_.push(batch);
	}

	template <typename Dtype>
	void ExBasePrefetchingDataLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		ExBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
		// Reshape to loaded data.
		top[0]->ReshapeLike(batch->data_);
		top[1]->ReshapeLike(batch->label_);
		top[2]->ReshapeLike(batch->dense_);
		// Copy the data
		caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
			top[0]->mutable_gpu_data());
		caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
			top[1]->mutable_gpu_data());
		caffe_copy(batch->dense_.count(), batch->dense_.gpu_data(),
			top[2]->mutable_gpu_data());
		// Ensure the copy is synchronous wrt the host, so that the next batch isn't
		// copied in meanwhile.
		CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
		prefetch_free_.push(batch);
	}

	template <typename Dtype>
	void RoiPrefetchingDataLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		RoiBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
		// Reshape to loaded data.
		top[0]->ReshapeLike(batch->data_);
		top[1]->ReshapeLike(batch->info_);
		if (output_labels_)
		{
			top[2]->ReshapeLike(batch->label_a_);
			top[3]->ReshapeLike(batch->label_c_);
			top[4]->ReshapeLike(batch->label_s_);
		}
		// Copy the data
		caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
			top[0]->mutable_gpu_data());
		caffe_copy(batch->info_.count(), batch->info_.gpu_data(),
			top[1]->mutable_gpu_data());
		if (output_labels_) {
			caffe_copy(batch->label_a_.count(), batch->label_a_.gpu_data(),
				top[2]->mutable_gpu_data());
			caffe_copy(batch->label_c_.count(), batch->label_c_.gpu_data(),
				top[3]->mutable_gpu_data());
			caffe_copy(batch->label_s_.count(), batch->label_s_.gpu_data(),
				top[4]->mutable_gpu_data());
		}
		// Ensure the copy is synchronous wrt the host, so that the next batch isn't
		// copied in meanwhile.
		CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
		prefetch_free_.push(batch);
	}

	INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
	INSTANTIATE_LAYER_GPU_FORWARD(ExBasePrefetchingDataLayer);
	INSTANTIATE_LAYER_GPU_FORWARD(RoiPrefetchingDataLayer);

}  // namespace caffe
