# Triple-branch Fully Convolutional Networks for Organ Localization in CT image
This is a modified version of [Caffe](https://github.com/BVLC/caffe) which supports the **3D Triple-branch Fully Convolutional Networks** as described in our paper [**Multiple Organ Localization in CT Image using Anisotropic Triple-branch Fully Convolutional Networks**] (under review).

<img src="./workflow.png"/>

This code has been compiled and passed on `Windows 7 (64 bits)` using `Visual Studio 2013`.

## How to build

**Requirements**: `Visual Studio 2013`, `ITK-4.10`, `CUDA 8.0` and `cuDNN v5`

### Pre-Build Steps
Please make sure CUDA and cuDNN have been installed correctly on your computer.

Clone the project by running:
```
git clone https://github.com/superxuang/caffe_triple-branch_FCN.git
```

In `.\Caffe.bat` set `ITK_PATH` to ITK intall path (the path containing ITK `include`,`lib` folders).

### Build
Run `.\Caffe.bat` and build the project `caffe` in `Visual Studio 2013`.

## How to use
### Download image data
Please download and unzip the CT images from [LiTS challenge](https://competitions.codalab.org/competitions/17094). Note that, the original CT images of LiTS dataset are stored in `*.nii` format. Please convert them to `*.mhd` format.

### Download annotation data
The organ bounding-box annotations could be downloaded from [this repository](./annotations_on_LiTS/) or [IEEE DataPort](http://dx.doi.org/10.21227/df8g-pq27).

### Prepare data
Move the CT images and the bounding-box annotations to a data folder, and create an entry list file (`_train_set_list.txt`) in the same folder. To this end, the data folder is organized in the folloing way:

```
└── data folder
    ├── _train_set_list.txt
    ├── volume-0.mhd
    ├── volume-0.raw
    ├── segmentation-0.mhd
    ├── segmentation-0.raw
    ├── segmentation-0.txt
    ├── volume-1.mhd
    ├── volume-1.raw
    ├── segmentation-1.mhd
    ├── segmentation-1.raw
    ├── segmentation-1.txt
    |   ....................... 
    ├── volume-130.mhd
    ├── volume-130.raw
    ├── segmentation-130.mhd
    ├── segmentation-130.raw
    └── segmentation-130.txt
```

The entry list file `_train_set_list.txt` stores the filenames that are actually used for training. Each line of the entry list file corresponds a data sample. Here is an example of the entry list file `_train_set_list.txt` corresponding to above data folder. **Note that, the segmentation mask file (`segmentation-N.mhd` and `segmentation-N.raw`) is not necessary for neither training nor testing. We just reserve it for further research (e.g. organ segmentation). You could just make the segmentation mask files absent and fill the second column of the entry list file with a non-existent filename.**  

```
volume-0.mhd segmentation-0.mhd segmentation-0.txt
volume-1.mhd segmentation-1.mhd segmentation-1.txt
.......................
volume-130.mhd segmentation-130.mhd segmentation-130.txt
```

### Start the training
Modify the path parameter of datalayer in `.\models\triple-branch_FCN\net.prototxt`.
```
layer {
  name: "mhd_input"
  type: "MHDRoiData"
  top: "data"
  top: "im_info"
  top: "label_a"
  top: "label_c"
  top: "label_s"
  mhd_data_param {  
    source: "F:/Deep/MyDataset/LITS/_train_set_list.txt" # the entry list file mentioned above  
    root_folder: "F:/Deep/MyDataset/LITS/" # the data folder mentioned above
    batch_size: 3
    shuffle: true
    hist_matching: false
    truncate_probability: 0.5
    min_truncate_length: 128
    inplane_shift: 5
    min_intensity: -1000
    max_intensity: 1600
    random_deform: 0.0
    deform_control_point: 2
    deform_sigma: 15.0
    contour_name_list {
      name: "liver"
      name: "lung-r"
      name: "lung-l"
      name: "kidney-r"
      name: "kidney-l"
      name: "femur-r"
      name: "femur-l"
      name: "bladder"
      name: "heart"
      name: "spleen"
      name: "pancreas"
    }
    resample_size_x: 256
    resample_size_y: 256
    resample_size_z: 512
  }
  include: { phase: TRAIN }
}
```
Run `.\models\triple-branch_FCN\train.bat`

## License and Citation

Please cite Caffe if it is useful for your research:

    @article{jia2014caffe,
      author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      journal = {arXiv preprint arXiv:1408.5093},
      title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      year = {2014}
    }
    
Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.