#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

	/**
	* @brief Provides base for data layers that feed blobs to the Net.
	*
	* TODO(dox): thorough documentation for Forward and proto params.
	*/
	template <typename Dtype>
	class BaseDataLayer : public Layer<Dtype> {
	public:
		explicit BaseDataLayer(const LayerParameter& param);
		// LayerSetUp: implements common data layer setup functionality, and calls
		// DataLayerSetUp to do special data layer setup for individual layer types.
		// This method may not be overridden except by the BasePrefetchingDataLayer.
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		// Data layers should be shared by multiple solvers in parallel
		virtual inline bool ShareInParallel() const { return true; }
		virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {}
		// Data layers have no bottoms, so reshaping is trivial.
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {}

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

	protected:
		TransformationParameter transform_param_;
		shared_ptr<DataTransformer<Dtype> > data_transformer_;
		bool output_labels_;
	};

	template <typename Dtype>
	class Batch {
	public:
		Blob<Dtype> data_, label_;
	};

	template <typename Dtype>
	class BasePrefetchingDataLayer :
		public BaseDataLayer<Dtype>, public InternalThread {
	public:
		explicit BasePrefetchingDataLayer(const LayerParameter& param);
		// LayerSetUp: implements common data layer setup functionality, and calls
		// DataLayerSetUp to do special data layer setup for individual layer types.
		// This method may not be overridden.
		void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		// Prefetches batches (asynchronously if to GPU memory)
		static const int PREFETCH_COUNT = 3;

	protected:
		virtual void InternalThreadEntry();
		virtual void load_batch(Batch<Dtype>* batch) = 0;

		Batch<Dtype> prefetch_[PREFETCH_COUNT];
		BlockingQueue<Batch<Dtype>*> prefetch_free_;
		BlockingQueue<Batch<Dtype>*> prefetch_full_;

		Blob<Dtype> transformed_data_;
	};

	template <typename Dtype>
	class ExBatch {
	public:
		Blob<Dtype> data_, label_, dense_;
	};

	template <typename Dtype>
	class ExBasePrefetchingDataLayer :
		public BaseDataLayer<Dtype>, public InternalThread {
	public:
		explicit ExBasePrefetchingDataLayer(const LayerParameter& param);
		// LayerSetUp: implements common data layer setup functionality, and calls
		// DataLayerSetUp to do special data layer setup for individual layer types.
		// This method may not be overridden.
		void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		// Prefetches batches (asynchronously if to GPU memory)
		static const int PREFETCH_COUNT = 3;

	protected:
		virtual void InternalThreadEntry();
		virtual void load_batch(ExBatch<Dtype>* batch) = 0;

		ExBatch<Dtype> prefetch_[PREFETCH_COUNT];
		BlockingQueue<ExBatch<Dtype>*> prefetch_free_;
		BlockingQueue<ExBatch<Dtype>*> prefetch_full_;

		Blob<Dtype> transformed_data_;
	};

	template <typename Dtype>
	class RoiBatch {
	public:
		Blob<Dtype> data_, info_, label_a_, label_c_, label_s_;
	};

	template <typename Dtype>
	class RoiPrefetchingDataLayer :
		public BaseDataLayer<Dtype>, public InternalThread {
	public:
		explicit RoiPrefetchingDataLayer(const LayerParameter& param);
		// LayerSetUp: implements common data layer setup functionality, and calls
		// DataLayerSetUp to do special data layer setup for individual layer types.
		// This method may not be overridden.
		void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		// Prefetches batches (asynchronously if to GPU memory)
		static const int PREFETCH_COUNT = 1;

	protected:
		virtual void InternalThreadEntry();
		virtual void load_batch(RoiBatch<Dtype>* batch) = 0;

		RoiBatch<Dtype> prefetch_[PREFETCH_COUNT];
		BlockingQueue<RoiBatch<Dtype>*> prefetch_free_;
		BlockingQueue<RoiBatch<Dtype>*> prefetch_full_;
	};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
