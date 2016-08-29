// Standard and Specific C++ headers

#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

// OpenCV headers

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// Caffe headers

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// caffe.proto > LayerParameter > WindowPoseDataParameter
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size

#if CV_VERSION_MAJOR == 3
const int CV_LOAD_IMAGE_COLOR = cv::IMREAD_COLOR;
#endif

namespace caffe {

// Destructor
template <typename Dtype>
WindowPoseDataLayerRegression<Dtype>::~WindowPoseDataLayerRegression<Dtype>() {
  this->JoinPrefetchThread();
}

// Does layer-specific setup
template <typename Dtype>
void WindowPoseDataLayerRegression<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // LayerSetUp runs through the window_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2 e1 e2 e3 e1c e2c e3c


  // Info to be printed to the LOG
  LOG(INFO) << "Window data layer:" << std::endl
      << "  foreground (object) overlap threshold: "
      << this->layer_param_.window_data_param().fg_threshold() << std::endl
      << "  background (non-object) overlap threshold: "
      << this->layer_param_.window_data_param().bg_threshold() << std::endl
      << "  foreground sampling fraction: "
      << this->layer_param_.window_data_param().fg_fraction() << std::endl
      << "  cache_images: "
      << this->layer_param_.window_data_param().cache_images() << std::endl
      << "  root_folder: "
      << this->layer_param_.window_data_param().root_folder();

  // Whether or not to cache images. If set to true, will load all images into memory, 
  // for faster access. Default value is false.
  cache_images_ = this->layer_param_.window_data_param().cache_images();
  // Whether or not to apped the root folder, to be able to find images.
  string root_folder = this->layer_param_.window_data_param().root_folder();

  // Whether or not randomization is needed, i.e., mirroring or random crops
  const bool prefetch_needs_rand = this->transform_param_.mirror() || this->transform_param_.crop_size();
  // If randomized prefetching is needed, generate a seed from Caffe's RNG.
  if (prefetch_needs_rand) {
    // Fetch a function reference Caffe's RNG
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }

  // Opens the source file specified in the window data parameter options.
  // The source file is the file containing all data in the prescribed format.
  std::ifstream infile(this->layer_param_.window_data_param().source().c_str());
  // Check if the state of the stream is good. Else, display and error message
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.window_data_param().source() << std::endl;

  // Create a string that is used to store a hashtag
  string hashtag;
  // Variables to store image index and number of channels
  int image_index, channels;
  // If nothing can be read into 'hashtag', throw an error stating that the file is empty.
  if (!(infile >> hashtag >> image_index)) {
    LOG(FATAL) << "Window file is empty";
  }
  // A new image usually begins with a hashtag
  do {
    CHECK_EQ(hashtag, "#");
    // Read image path
    string image_path;
    infile >> image_path;
    // If root folder is to be appended (as per options specified in WindowLayerParameters), append it.
    image_path = root_folder + image_path;
    // Read in image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2];
    channels = image_size[0];
    // Add the (image path, image size) pair to the image database
    image_database_.push_back(std::make_pair(image_path, image_size));


    // If image is to be loaded into memory, create a hashmap for the (image path, image datum)
    if (cache_images_) {
      Datum datum;
      if (!ReadFileToDatum(image_path, &datum)) {
        LOG(ERROR) << "Could not open or find file " << image_path;
        return;
      }
      image_database_cache_.push_back(std::make_pair(image_path, datum));
    }
    // Read each window
    int num_windows;
    infile >> num_windows;
    // Get foreground and background thresholds
    const float fg_threshold = this->layer_param_.window_data_param().fg_threshold();
    const float bg_threshold = this->layer_param_.window_data_param().bg_threshold();
    
    // For each window present in the current image, read in data and add it to the fg_windows vector
    for (int i = 0; i < num_windows; ++i) {

      // Bounding box coordinates
      int x1, y1, x2, y2;
      // Labels for azimuth and elevation
      float sin_azimuth, cos_azimuth, sin_elevation, cos_elevation;
      // Read the data
      infile >> x1 >> y1 >> x2 >> y2 >> sin_azimuth >> cos_azimuth >> sin_elevation >> cos_elevation;

      // Create a window vector
      vector<float> window(WindowPoseDataLayerRegression::NUM);
      // Store read data in the vector
      window[WindowPoseDataLayerRegression::IMAGE_INDEX] = image_index;
      window[WindowPoseDataLayerRegression::X1] = x1;
      window[WindowPoseDataLayerRegression::Y1] = y1;
      window[WindowPoseDataLayerRegression::X2] = x2;
      window[WindowPoseDataLayerRegression::Y2] = y2;
      window[WindowPoseDataLayerRegression::SIN_AZIMUTH] = sin_azimuth;
      window[WindowPoseDataLayerRegression::COS_AZIMUTH] = cos_azimuth;
      window[WindowPoseDataLayerRegression::SIN_ELEVATION] = sin_elevation;
      window[WindowPoseDataLayerRegression::COS_ELEVATION] = cos_elevation; 

      // Add the window to the foreground list
      fg_windows_.push_back(window);
    }

    // After every 100 images processed, print data to the log
    if (image_index % 100 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_windows;
    }
  // If the next value is exists, run the loop once again
  } while (infile >> hashtag >> image_index);

  // Print the number of images processed
  LOG(INFO) << "Number of images: " << image_index+1;

  // Display the amount of context padding added to the window (usually 0)
  LOG(INFO) << "Amount of context padding: "
      << this->layer_param_.window_data_param().context_pad();

  // Specify the crop mode (warp vs square) used (usually warp)
  LOG(INFO) << "Crop mode: "
      << this->layer_param_.window_data_param().crop_mode();

  // Read image and write it to the first top blob
  const int crop_size = this->transform_param_.crop_size();
  // Ensure that the crop size is greater than zero
  CHECK_GT(crop_size, 0);

  // Get batch size
  const int batch_size = this->layer_param_.window_data_param().batch_size();
  // Reshape the top blob and the prefetch_data_ blob accordingly
  top[0]->Reshape(batch_size, channels, crop_size, crop_size);
  this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);

  // Print output data dimensions
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // Do not output labels - Added by KM
  // If the following line isn't present, the data layer attempts to output labels,
  // and throws an error if labels are not present in the window data file.
  this->output_labels_ = false;

  // Label blobs, in the order sin_azimuth, cos_azimuth, sin_elevation, cos_elevation
  // top[1]->Reshape(batch_size, 1, 1, 1);
  // top[2]->Reshape(batch_size, 1, 1, 1);
  // top[3]->Reshape(batch_size, 1, 1, 1);
  // top[4]->Reshape(batch_size, 1, 1, 1);
  top[1]->Reshape(batch_size, 4, 1, 1);
  // Reshape the data prefetch blobs
  this->prefetch_sin_azimuth_.Reshape(batch_size, 1, 1, 1);
  this->prefetch_cos_azimuth_.Reshape(batch_size, 1, 1, 1);
  this->prefetch_sin_elevation_.Reshape(batch_size, 1, 1, 1);
  this->prefetch_cos_elevation_.Reshape(batch_size, 1, 1, 1);
  // Reshape the label blobs
  this->prefetch_labels_.Reshape(batch_size, 4, 1, 1);

  // Whether mean file or mean values are specified (in our case, mean values are given)
  has_mean_file_ = this->transform_param_.has_mean_file();
  has_mean_values_ = this->transform_param_.mean_value_size() > 0;
  if (has_mean_file_) {
    const string& mean_file =
          this->transform_param_.mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // If mean file is not specified, use these mean values per channel for subtraction
  float meanVals[3] = {102.98,115.95,122.77};
  if (1) {
    // Ensure that both mean file and mean values are not specified
    CHECK(has_mean_file_ == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    // Read in the channnel-wise mean values
    for (int c = 0; c < 3; ++c) {
      mean_values_.push_back(meanVals[c]);
    }
    // Check that as many mean values as required are specified (we need one per channel)
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
     "Specify either 1 mean_value or as many as channels: " << channels;
    // Replicate mean values if only 1 mean value is specified
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
}

// Something to do with randomization during the prefetch phase (???)
template <typename Dtype>
unsigned int WindowPoseDataLayerRegression<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}


// Thread that fetching the data
template <typename Dtype>
void WindowPoseDataLayerRegression<Dtype>::InternalThreadEntry() {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  // --KM-- We have only foreground windows, for now

  // Start timer for the batch
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  // Declare another timer
  CPUTimer timer;
  // Data (image as datum)
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  // // Get sin_azimuth from the prefetch thread
  // Dtype* top_sin_azimuth = this->prefetch_sin_azimuth_.mutable_cpu_data();
  // // Get cos_azimuth from the prefetch thread
  // Dtype* top_cos_azimuth = this->prefetch_cos_azimuth_.mutable_cpu_data();
  // // Get sin_elevation from the prefetch thread
  // Dtype* top_sin_elevation = this->prefetch_sin_elevation_.mutable_cpu_data();
  // // Get cos_elevation from the prefetch thread
  // Dtype* top_cos_elevation = this->prefetch_cos_elevation_.mutable_cpu_data();
  // Get labels from the prefetch thread
  Dtype* top_labels = this->prefetch_labels_.mutable_cpu_data();

  // Check whether or not scaling of data has to be done
  const Dtype scale = this->layer_param_.window_data_param().scale();
  // Get batch size
  const int batch_size = this->layer_param_.window_data_param().batch_size();
  // Get context padding
  const int context_pad = this->layer_param_.window_data_param().context_pad();
  // Get crop size
  const int crop_size = this->transform_param_.crop_size();
  // Whether or not to mirror a few samples randomly
  const bool mirror = this->transform_param_.mirror();

  // Variable declared to store the mean file, if present
  Dtype* mean = NULL;
  int mean_off = 0;
  int mean_width = 0;
  int mean_height = 0;
  if (this->has_mean_file_) {
    mean = this->data_mean_.mutable_cpu_data();
    mean_off = (this->data_mean_.width() - crop_size) / 2;
    mean_width = this->data_mean_.width();
    mean_height = this->data_mean_.height();
  }
  // Specify crop size and crop mode
  cv::Size cv_crop_size(crop_size, crop_size);
  const string& crop_mode = this->layer_param_.window_data_param().crop_mode();
  // Whether or not to use the square crop mode (usually false)
  bool use_square = (crop_mode == "square") ? true : false;

  // Initialize the batch with a zero-blob of appropriate dimensionality
  caffe_set(this->prefetch_data_.count(), Dtype(0), top_data);

  // ### Delete this!
  // const int num_fg = static_cast<int>(static_cast<float>(batch_size)
  //     * fg_fraction);
  // const int num_samples[2] = { batch_size - num_fg, num_fg };

  // Batch size
  const int num_samples = batch_size;

  // Index of item in the current blob
  int item_id = 0;

  // For each sample in the current batch
  for (int idx = 0; idx < num_samples; ++idx) {
    // Start timer
    timer.Start();
    // Sample a window randomly
    const unsigned int rand_index = PrefetchRand();
    std::vector<float> window = fg_windows_[rand_index % fg_windows_.size()];

    // If mirror is set to true, randomly mirror (we won't do it)
    // bool do_mirror = mirror && PrefetchRand() % 2;

    // Load the image containing the window
    pair<std::string, vector<int> > image =
      image_database_[window[WindowPoseDataLayerRegression<Dtype>::IMAGE_INDEX]];

    // Create a Mat variable to hold the image
    cv::Mat cv_img;
    // Load the image
    if (this->cache_images_) {
      pair<std::string, Datum> image_cached =
      image_database_cache_[window[WindowPoseDataLayerRegression<Dtype>::IMAGE_INDEX]];
      cv_img = DecodeDatumToCVMat(image_cached.second);
    } 
    else {
      cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
      if (!cv_img.data) {
        LOG(ERROR) << "Could not open or find file " << image.first;
        return;
      }
    }
    read_time += timer.MicroSeconds();

    timer.Start();
    const int channels = cv_img.channels();

    // Crop window out of image and warp it
    int x1 = window[WindowPoseDataLayerRegression<Dtype>::X1];
    int y1 = window[WindowPoseDataLayerRegression<Dtype>::Y1];
    int x2 = window[WindowPoseDataLayerRegression<Dtype>::X2];
    int y2 = window[WindowPoseDataLayerRegression<Dtype>::Y2];

    int pad_w = 0;
    int pad_h = 0;
    if (context_pad > 0 || use_square) {
      // scale factor by which to expand the original region
      // such that after warping the expanded region to crop_size x crop_size
      // there's exactly context_pad amount of padding on each side
      Dtype context_scale = static_cast<Dtype>(crop_size) /
        static_cast<Dtype>(crop_size - 2*context_pad);

      // compute the expanded region
      Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
      Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
      Dtype center_x = static_cast<Dtype>(x1) + half_width;
      Dtype center_y = static_cast<Dtype>(y1) + half_height;
      if (use_square) {
        if (half_height > half_width) {
          half_width = half_height;
        } else {
          half_height = half_width;
        }
      }
      x1 = static_cast<int>(round(center_x - half_width*context_scale));
      x2 = static_cast<int>(round(center_x + half_width*context_scale));
      y1 = static_cast<int>(round(center_y - half_height*context_scale));
      y2 = static_cast<int>(round(center_y + half_height*context_scale));

      // the expanded region may go outside of the image
      // so we compute the clipped (expanded) region and keep track of
      // the extent beyond the image
      int unclipped_height = y2-y1+1;
      int unclipped_width = x2-x1+1;
      int pad_x1 = std::max(0, -x1);
      int pad_y1 = std::max(0, -y1);
      int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
      int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
      // clip bounds
      x1 = x1 + pad_x1;
      x2 = x2 - pad_x2;
      y1 = y1 + pad_y1;
      y2 = y2 - pad_y2;
      CHECK_GT(x1, -1);
      CHECK_GT(y1, -1);
      CHECK_LT(x2, cv_img.cols);
      CHECK_LT(y2, cv_img.rows);

      int clipped_height = y2-y1+1;
      int clipped_width = x2-x1+1;

      // scale factors that would be used to warp the unclipped
      // expanded region
      Dtype scale_x =
        static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_width);
      Dtype scale_y =
        static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_height);

      // size to warp the clipped expanded region to
      cv_crop_size.width =
        static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
      cv_crop_size.height =
        static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
      pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
      pad_x2 = static_cast<int>(round(static_cast<Dtype>(pad_x2)*scale_x));
      pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
      pad_y2 = static_cast<int>(round(static_cast<Dtype>(pad_y2)*scale_y));

      pad_h = pad_y1;
      // if we're mirroring, we mirror the padding too (to be pedantic)
      pad_w = pad_x1;

      // ensure that the warped, clipped region plus the padding fits in the
      // crop_size x crop_size image (it might not due to rounding)
      if (pad_h + cv_crop_size.height > crop_size) {
        cv_crop_size.height = crop_size - pad_h;
      }
      if (pad_w + cv_crop_size.width > crop_size) {
        cv_crop_size.width = crop_size - pad_w;
      }
    }

    cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
    cv::Mat cv_cropped_img = cv_img(roi);
    cv::resize(cv_cropped_img, cv_cropped_img,
        cv_crop_size, 0, 0, cv::INTER_LINEAR);

    // Copy the warped window into top_data
    for (int h = 0; h < cv_cropped_img.rows; ++h) {
      const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < cv_cropped_img.cols; ++w) {
        for (int c = 0; c < channels; ++c) {
          int top_index = ((item_id * channels + c) * crop_size + h + pad_h)
            * crop_size + w + pad_w;
          // int top_index = (c * height + h) * width + w;
          Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
          if (this->has_mean_file_) {
            int mean_index = (c * mean_height + h + mean_off + pad_h)
              * mean_width + w + mean_off + pad_w;
            top_data[top_index] = (pixel - mean[mean_index]) * scale;
          } else {
            if (1) {
              top_data[top_index] = (pixel - this->mean_values_[c]) * scale;
            } else {
              top_data[top_index] = pixel * scale;
            }
          }
        }
      }
    }
  
    trans_time += timer.MicroSeconds();

    // Get Labels for the current window
    // top_sin_azimuth[idx] = window[WindowPoseDataLayerRegression<Dtype>::SIN_AZIMUTH];
    // top_cos_azimuth[idx] = window[WindowPoseDataLayerRegression<Dtype>::COS_AZIMUTH];
    // top_sin_elevation[idx] = window[WindowPoseDataLayerRegression<Dtype>::SIN_ELEVATION];
    // top_cos_elevation[idx] = window[WindowPoseDataLayerRegression<Dtype>::COS_ELEVATION];
    top_labels[idx*4 + 0] = window[WindowPoseDataLayerRegression<Dtype>::SIN_AZIMUTH];
    top_labels[idx*4 + 1] = window[WindowPoseDataLayerRegression<Dtype>::COS_AZIMUTH];
    top_labels[idx*4 + 2] = window[WindowPoseDataLayerRegression<Dtype>::SIN_ELEVATION];
    top_labels[idx*4 + 3] = window[WindowPoseDataLayerRegression<Dtype>::COS_ELEVATION];
  }

  batch_timer.Stop();

}

// Defines the forward pass using CPU
template <typename Dtype>
void WindowPoseDataLayerRegression<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  this->JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
            top[0]->mutable_cpu_data());
  caffe_copy(this->prefetch_labels_.count(), this->prefetch_labels_.cpu_data(),
    top[1]->mutable_cpu_data());
  // caffe_copy(this->prefetch_sin_azimuth_.count(), this->prefetch_sin_azimuth_.cpu_data(),
  //           top[1]->mutable_cpu_data());
  // caffe_copy(this->prefetch_cos_azimuth_.count(), this->prefetch_cos_azimuth_.cpu_data(),
  //           top[2]->mutable_cpu_data());
  // caffe_copy(this->prefetch_sin_elevation_.count(), this->prefetch_sin_elevation_.cpu_data(),
  //           top[3]->mutable_cpu_data());
  // caffe_copy(this->prefetch_cos_elevation_.count(), this->prefetch_cos_elevation_.cpu_data(),
  //           top[4]->mutable_cpu_data());
  // Start a new prefetch thread
  DLOG(INFO) << "Prefetch copied";
  DLOG(INFO) << "CreatePrefetchThread";
  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(WindowPoseDataLayerRegression);
REGISTER_LAYER_CLASS(WINDOW_POSE_DATA_REGRESSION, WindowPoseDataLayerRegression);
}  // namespace caffe
