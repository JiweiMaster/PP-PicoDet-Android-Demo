# NCNN
The PP-Picodet Object Detection Base on [NCNN](https://github.com/Tencent/ncnn) and opencv(https://github.com/opencv/opencv/releases/download/4.5.4/opencv-4.5.4-android-sdk.zip)


## APK Download
[app download link](https://paddledet.bj.bcebos.com/deploy/third_engine/PP-PicoDet.apk)


# Quick Start
## step1
* Download ncnn-YYYYMMDD-android-vulkan.zip or build ncnn for android yourself
* Extract ncnn-YYYYMMDD-android-vulkan.zip into `app/src/main/jni` and change the `ncnn_DIR` path to yours in `app/src/main/jni/CMakeLists.txt`

## step2
* Download [opencv-android](https://github.com/opencv/opencv/releases/download/4.5.4/opencv-4.5.4-android-sdk.zip)
* Extract opencv-4.5.4-android-sdk.zip into `app/src/main/jni` and change the `OpenCV_DIR` path to yours in `app/src/main/jni/CMakeLists.txt`

## step3
* Open this project with Android Studio, build it and enjoy!

# Notes
* Android ndk camera is used for best efficiency
* Crash may happen on very old devices for lacking HAL3 camera interface
* All models are manually modified to accept dynamic input shape
* Most small models run slower on GPU than on CPU, this is common
* FPS may be lower in dark environment because of longer camera exposure time

# Demo
<div>
<img src="./picodet-small.gif" width="40%" height="40%"/>
<img src="./nanodet-small.gif" width="40%" height="40%"/>
</div>
