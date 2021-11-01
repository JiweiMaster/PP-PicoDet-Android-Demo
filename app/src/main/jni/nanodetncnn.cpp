// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>
#include <platform.h>
#include <benchmark.h>
#include "nanodet.h"
#include "ndkcamera.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video.hpp>
#include <ctime>
#include <iostream>
#include <opencv2/videoio/legacy/constants_c.h>

void delay(int time){
//    clock_t now = ncnn:getc();
//    while(clock() - now < time);
    double now = ncnn::get_current_time();
    while(ncnn::get_current_time() - now < time);

}

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#define fps_length 60

#define mylog(...) __android_log_print
using namespace cv;
static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static float fps_history[fps_length] = {0.f};

//float prob_threshold = 0.3;
//float nms_threshold = 0.3;

static void clear_Fps_History(){
    for(int i=0; i<fps_length; i++){
        fps_history[i] = 0.f;
    }
}

static int draw_fps(cv::Mat& rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = fps_length-1; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[fps_length-1] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < fps_length; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= fps_length;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

//static int draw_fps(cv::Mat& rgb)
//{
//    // resolve moving average
//    float avg_fps = 0.f;
//    {
//        static double t0 = 0.f;
//        static float fps_history[10] = {0.f};
//
//        double t1 = ncnn::get_current_time();
//        if (t0 == 0.f)
//        {
//            t0 = t1;
//            return 0;
//        }
//
//        float fps = 1000.f / (t1 - t0);
//        t0 = t1;
//
//        for (int i = 9; i >= 1; i--)
//        {
//            fps_history[i] = fps_history[i - 1];
//        }
//        fps_history[0] = fps;
//
//        if (fps_history[9] == 0.f)
//        {
//            return 0;
//        }
//
//        for (int i = 0; i < 10; i++)
//        {
//            avg_fps += fps_history[i];
//        }
//        avg_fps /= 10.f;
//    }
//
//    char text[32];
//    sprintf(text, "FPS=%.2f", avg_fps);
//
//    int baseLine = 0;
//    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//
//    int y = 0;
//    int x = rgb.cols - label_size.width;
//
//    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
//                    cv::Scalar(255, 255, 255), -1);
//
//    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
//                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
//
//    return 0;
//}

static NanoDet* g_nanodet = 0;
static ncnn::Mutex lock;

class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    // nanodet
    {
        ncnn::MutexLockGuard g(lock);
        if (g_nanodet)
        {
            std::vector<Object> objects;
            if(g_nanodet->mode_type == 0){
                g_nanodet->detect(rgb, objects);
            }
            if(g_nanodet->mode_type == 1){
                g_nanodet->detectPicoDet(rgb, objects);
            }
            if(g_nanodet->mode_type == 2){
                g_nanodet->detectPicoDetFourHead(rgb, objects);
            }
            if(g_nanodet->mode_type == 3){
                g_nanodet->detectYolox(rgb, objects);
            }
            if(g_nanodet->mode_type == 4){
                g_nanodet->detectYoloV5(rgb, objects);
            }
            g_nanodet->draw(rgb, objects);
        }
        else
        {
            draw_unsupported(rgb);
        }
    }

    draw_fps(rgb);
}
// 无法生成图片，可能和opencv的有关
void genVideoByImg(){
//    /storage/emulated/0/Android/data/com.baidu.picodetncnn/cache/out_infer.avi
    VideoWriter videoWriter("out_infer.avi",
                            CV_FOURCC('M', 'J', 'P', 'G'),
                            10.0, Size(640, 640));
    for(int frameNum=1;frameNum<61;frameNum++){
        char imgName[100] = {};
        sprintf(imgName,"/storage/emulated/0/Android/data/com.baidu.picodetncnn/cache/%d.jpg",frameNum);
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn:  images","%s", imgName);
        cv::Mat img = imread(imgName);
        videoWriter<<img;
        delay(100);
    }
    videoWriter.release();
}

double read_img_time = 0;
double draw_img_time = 0;
double detect_time = 0;
double imwrite_time=0;

double analysis_cooc128() {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn: ", "%s", "analysis_cooc128");
    cv::VideoCapture inputVideo("/storage/emulated/0/Android/data/com.baidu.picodetncnn/cache/out.avi");
    double start_time = ncnn::get_current_time();
    int frameNum = 0;
    unsigned int objectsNum=0;
    while(true){
        double time_idx = ncnn::get_current_time();
        frameNum++;
        cv::Mat rgb;
        cv::Mat rgb_in;
        inputVideo>>rgb_in;
        if(rgb_in.empty()){
            break;
        }
        cvtColor(rgb_in, rgb, COLOR_BGR2RGB);
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn: frame number", "%d", frameNum);
        char imgName[1000] = {};
        if (g_nanodet) {
            std::vector<Object> objects;
            if (g_nanodet->mode_type == 0) { // nanodet head
                read_img_time = read_img_time + (ncnn::get_current_time() - time_idx);
                time_idx = ncnn::get_current_time();
                g_nanodet->detect(rgb, objects);
                detect_time = detect_time + (ncnn::get_current_time()-time_idx);
                objectsNum = objectsNum + objects.size();
                time_idx = ncnn::get_current_time();
                g_nanodet->draw(rgb, objects);
                draw_img_time = draw_img_time + (ncnn::get_current_time()-time_idx);
            }
            if (g_nanodet->mode_type == 1) { // picodet three head
                g_nanodet->detectPicoDet(rgb, objects);
                objectsNum = objectsNum + objects.size();
                g_nanodet->draw(rgb, objects);
            }
            if (g_nanodet->mode_type == 2) { // pciodet four head
                read_img_time = read_img_time + (ncnn::get_current_time() - time_idx);
                time_idx = ncnn::get_current_time();
                g_nanodet->detectPicoDetFourHead(rgb, objects, 0.45,0.5);
                detect_time = detect_time + (ncnn::get_current_time()-time_idx);
                objectsNum = objectsNum + objects.size();
                time_idx = ncnn::get_current_time();
                g_nanodet->draw(rgb, objects);
                draw_img_time = draw_img_time + (ncnn::get_current_time()-time_idx);
            }
            if (g_nanodet->mode_type == 3) {
                read_img_time = read_img_time + (ncnn::get_current_time() - time_idx);
                time_idx = ncnn::get_current_time();
                g_nanodet->detectYolox(rgb, objects);
                detect_time = detect_time + (ncnn::get_current_time()-time_idx);
                objectsNum = objectsNum + objects.size();
                time_idx = ncnn::get_current_time();
                g_nanodet->draw(rgb, objects);
                draw_img_time = draw_img_time + (ncnn::get_current_time()-time_idx);
            }
        }

        time_idx = ncnn::get_current_time();
        sprintf(imgName,"/storage/emulated/0/Android/data/com.baidu.picodetncnn/cache/%d.jpg",frameNum);
        cv::imwrite(imgName, rgb);
        imwrite_time = imwrite_time + (ncnn::get_current_time() - time_idx);
    }
    double stop_time = ncnn::get_current_time();
    return stop_time-start_time;
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn:", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn:", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);
        delete g_nanodet;
        g_nanodet = 0;
    }
    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_baidu_picodetncnn_NanoDetNcnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu)
{
    int curent_model = -1;
    if (modelid < 0 || modelid > 7 || cpugpu < 0 || cpugpu > 1)
    {
        return JNI_FALSE;
    }
    if(modelid == 0 || modelid == 1){
        curent_model = 0; //nanodet-m
    }
    if(modelid == 2){
        curent_model = 2; //picodet-s
    }
    if(modelid == 3){
        curent_model = 3; //yolox-nano
    }
    if(modelid == 4){
        curent_model = 4; //yolov5
    }
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    const char* modeltypes[] =
    {
        "nanodet-m-320",
        "nanodet-m-320",
        "picodet-s-320",
        "yolox-nano-320",
        "yolov5s",
    };
    const int target_sizes[] =
    {
        320,
        320,
        320,
        320,
        320,
    };
    const float mean_vals[][3] =
    {
        {103.53f, 116.28f, 123.675f},
        {103.53f, 116.28f, 123.675f},
        {103.53f, 116.28f, 123.675f},
        {103.53f, 116.28f, 123.675f},
        {103.53f, 116.28f, 123.675f}
    };
    const float norm_vals[][3] =
    {
        {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f},
        {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f},
        {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f},
        {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f},
        {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f}
    };
    const char* modeltype = modeltypes[(int)modelid];
    int target_size = target_sizes[(int)modelid];
    bool use_gpu = (int)cpugpu == 1;
    // reload
    {
        ncnn::MutexLockGuard g(lock);
        if (use_gpu && ncnn::get_gpu_count() == 0)
        {
            // no gpu
            delete g_nanodet;
            g_nanodet = 0;
        }
        else
        {
            if (!g_nanodet)
                g_nanodet = new NanoDet;
            g_nanodet->mode_type = curent_model;
            g_nanodet->load(mgr, modeltype, target_size, mean_vals[(int)modelid], norm_vals[(int)modelid], use_gpu);
        }
    }
    clear_Fps_History();
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn: current use model", "%s", modeltypes[g_nanodet->mode_type]);
    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_baidu_picodetncnn_NanoDetNcnn_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);
    g_camera->open((int)facing);
    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_baidu_picodetncnn_NanoDetNcnn_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");
    g_camera->close();
    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_baidu_picodetncnn_NanoDetNcnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

JNIEXPORT jdouble JNICALL Java_com_baidu_picodetncnn_NanoDetNcnn_testInferTime(JNIEnv *env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu) {
    int curent_model = -1;
    if (modelid < 0 || modelid > 7 || cpugpu < 0 || cpugpu > 1)
    {
        return JNI_FALSE;
    }
    if(modelid == 0 || modelid == 1){
        curent_model = 0; //nanodet
    }
    if(modelid == 2){
        curent_model = 2;//picodet-fourhead
    }
    if(modelid == 3){
        curent_model = 3;//yolox-nano
    }
    if(modelid == 4){
        curent_model = 4; //yolov5
    }
    const char* modeltypes[] = {
            "nanodet-m-320",
            "nanodet-m-320",
            "picodet-s-320",
            "yolox-nano-320",
            "yolov5s",
    };

    const int target_sizes[] ={320,320,320,320,320};

    const float mean_vals[][3] =
            {
                    {103.53f, 116.28f, 123.675f},
                    {103.53f, 116.28f, 123.675f},
                    {103.53f, 116.28f, 123.675f},
                    {103.53f, 116.28f, 123.675f},
                    {103.53f, 116.28f, 123.675f}
            };

    const float norm_vals[][3] =
            {
                    {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f},
                    {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f},
                    {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f},
                    {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f},
                    {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f}
            };
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    const char* modeltype = modeltypes[(int)modelid];
    int target_size = target_sizes[(int)modelid];
    bool use_gpu = (int)cpugpu == 1;
    // reload
    {
        ncnn::MutexLockGuard g(lock);
        if (use_gpu && ncnn::get_gpu_count() == 0)
        {
            // no gpu
            delete g_nanodet;
            g_nanodet = 0;
        }
        else
        {
            if (!g_nanodet)
                g_nanodet = new NanoDet;
            g_nanodet->mode_type = curent_model;
            g_nanodet->load(mgr, modeltype, target_size, mean_vals[(int)modelid], norm_vals[(int)modelid], use_gpu);
        }
    }
//  init params
    g_nanodet->postprocess_time_picodet=0;
    g_nanodet->infer_time_picodet=0;
    g_nanodet->preprocess_time_picodet=0;
    read_img_time = 0;
    draw_img_time = 0;
    detect_time = 0;
    imwrite_time = 0;
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn: current use model", "%s", modeltypes[g_nanodet->mode_type]);
//    double infertime1 = analysis_cooc128();
//    double infertime2 = analysis_cooc128();
//    double infertime3 = analysis_cooc128();
//    double infertime = (infertime1+infertime2+infertime3)/3;
    double infertime = analysis_cooc128();
    double process_time = g_nanodet->preprocess_time_picodet;
    double infer_time = g_nanodet->infer_time_picodet;
    double postprocess_time = g_nanodet->postprocess_time_picodet;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn: custome time:", "%lf", infertime);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn: process_time:", "%lf", process_time);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn: infer_time:", "%lf", infer_time);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn: postprocess_time:", "%lf", postprocess_time);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn: read_img_time:", "%lf", read_img_time);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn: draw_img_time:", "%lf", draw_img_time);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn: detect_time:", "%lf", detect_time);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn: imwrite_time:", "%lf", imwrite_time);
    return infertime;
}

JNIEXPORT jboolean JNICALL Java_com_baidu_picodetncnn_NanoDetNcnn_genVideoByImg(JNIEnv *env, jobject thiz) {
    genVideoByImg();
}
}

