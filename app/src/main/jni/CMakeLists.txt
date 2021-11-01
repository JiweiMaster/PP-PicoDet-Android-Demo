#project(picodetncnn)
#cmake_minimum_required(VERSION 3.10)
#set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv/sdk/native/jni)
##set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/aaa/sdk/native/jni)
##set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.3-android/sdk/native/jni)
##set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv/sdk/native/jni)
## jiwei
##include_directories(${CMAKE_SOURCE_DIR}/opencv/sdk/native/jni/include)
#find_package(OpenCV REQUIRED core imgproc highgui)
#set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20210720-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
#find_package(ncnn REQUIRED)
#add_library(picodetncnn SHARED nanodetncnn.cpp nanodet.cpp ndkcamera.cpp)
#target_link_libraries(picodetncnn ncnn ${OpenCV_LIBS} camera2ndk mediandk)




#project(picodetncnn)
##include(ndk-stl-config.cmake)
#cmake_minimum_required(VERSION 3.10)
#
## add name==picodetncnn, source file can be denoted
#include_directories(${CMAKE_SOURCE_DIR}/opencv/native/jni/include)
##add_library(lib_opencv SHARED IMPORTED)
##set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/opencv/native/libs/${ANDROID_ABI}/libopencv_java4.so)
#
## ncnn
#set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20210720-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
#find_package(ncnn REQUIRED)
#
## picodetncnn
#add_library(picodetncnn SHARED nanodetncnn.cpp nanodet.cpp ndkcamera.cpp)
## 被依赖的库放在依赖的库后面
##target_link_libraries(picodetncnn ncnn camera2ndk mediandk lib_opencv)
#
#target_link_libraries(picodetncnn ncnn camera2ndk mediandk)
#
#
#
#
#

project(picodetncnn)
cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv/sdk/native/jni)
#set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/OpenCV-android-sdk/sdk/native/jni)
#find_package(OpenCV REQUIRED core imgproc video videoio highgui)
find_package(OpenCV REQUIRED)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20210720-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)
add_library(picodetncnn SHARED nanodetncnn.cpp nanodet.cpp ndkcamera.cpp)
target_link_libraries(picodetncnn ncnn ${OpenCV_LIBS} camera2ndk mediandk)