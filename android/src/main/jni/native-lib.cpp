#include <jni.h> 
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <android/native_window_jni.h>
#include <android/log.h>
#include <thread>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <gst/gst.h>
#include <gst/video/videooverlay.h>  // Make sure this is included
#include <gst/app/gstappsink.h>
#include <gst/base/gstbasesink.h>
#include <ctime>
#include <chrono>
#include <gst/gstbin.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>  // Ensure Scalar is included
#include <future>

#define LOG_TAG "OpenCVAndroid"
GST_DEBUG_CATEGORY_STATIC (debug_category);
#define GST_CAT_DEFAULT debug_category
double downSample = 1.0;
int zoomFactor = 1.1;
double processVar = 0.03;
double measVar = 2;
int showFullScreen = 0;
int delay_time = 1;
double roiDiv = 4.0;
int showrectROI = 0;
int showTrackingPoints = 0;
int showUnstabilized = 0;
int maskFrame = 0;
std::mutex mtx;
std::condition_variable cv1;
bool ready1 = false;
bool ready2 = false;
bool ready3 = false;
bool ready4 = true;
cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03);
cv::Size subPixWinSize(10, 10), winSize(31, 31);
cv::Mat currFrame, currGray;
cv::Mat prevFrame, prevGray, prevOrig;
cv::Mat Orig, f_stabilized, fS;
cv::Mat X_estimate, P_estimate;
cv::Mat_<double> Q(1, 3, processVar);
cv::Mat_<double> R(1, 3, measVar);
std::vector<cv::Mat> K_collect;
std::vector<cv::Mat> P_collect;
cv::Mat lastRigidTransform = cv::Mat::eye(2, 3, CV_64F);
int count = 0;
int count1 = 0;
double x = 0, y = 0, a = 0;
cv::Mat prevFrame1, prevGray1, prevOrig1;
cv::Mat currFrame1, currGray1, currOrig1;
std::mutex mtx2;
std::vector<std::future<cv::Mat>> futures;
std::condition_variable odd_cv;
std::condition_variable even_cv;
std::atomic<bool> processing_active(true);
int num_cores_processing = -1;
int num = 0;
cv::Mat T;
std::atomic<bool> nativeflag(false);

#define LOG_TAG "OpenCVAndroid"
#if GLIB_SIZEOF_VOID_P == 8
# define GET_CUSTOM_DATA(env, thiz, fieldID) (CustomData *)(*env)->GetLongField (env, thiz, fieldID)
# define SET_CUSTOM_DATA(env, thiz, fieldID, data) (*env)->SetLongField (env, thiz, fieldID, (jlong)data)
#else
# define GET_CUSTOM_DATA(env, thiz, fieldID) (CustomData *)(jint)(*env)->GetLongField (env, thiz, fieldID)
# define SET_CUSTOM_DATA(env, thiz, fieldID, data) (*env)->SetLongField (env, thiz, fieldID, (jlong)(jint)data)
#endif
#define SET_CUSTOM_DATA(env, thiz, fieldID, data) \
    (env)->SetLongField((thiz), (fieldID), (jlong)(data))
#define GET_CUSTOM_DATA(env, thiz, fieldID) \
    (CustomData *)(env->GetLongField((thiz), (fieldID)))

typedef struct _CustomData
{
    jobject app;                  
    GstElement *pipeline;         
    GMainContext *context;        
    GMainLoop *main_loop;       
    gboolean initialized;        
    GstElement *video_sink;       
    ANativeWindow *native_window; 
    int incomingwidth;
    int incomingheight;
    int prev_width;
    int prev_height;
    gint64 last_pts;          // Last presentation timestamp
    bool first_frame;         // Flag for first frame
    int frame_count;          // Counter for frames
    std::chrono::steady_clock::time_point last_frame_time;
} CustomData;

static pthread_t gst_app_thread;
static pthread_key_t current_jni_env;
static JavaVM *java_vm;
static jfieldID custom_data_field_id = NULL;
static jmethodID set_message_method_id;
static jmethodID on_gstreamer_initialized_method_id;

static JNIEnv *attach_current_thread() {
    if (java_vm == NULL) {
        GST_ERROR("java_vm is NULL");
        return NULL;
    }

    JNIEnv *env;
    JavaVMAttachArgs args;

    GST_DEBUG("Attaching thread %p", g_thread_self());
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Attaching thread %p", g_thread_self());

    args.version = JNI_VERSION_1_4;
    args.name = NULL;
    args.group = NULL;

    if (java_vm->AttachCurrentThread(&env, &args) < 0) {
        GST_ERROR("Failed to attach current thread");
        return NULL;
    }

    return env;
}    static void
    detach_current_thread(void *env) {
        GST_DEBUG ("Detaching thread %p", g_thread_self());
        java_vm->DetachCurrentThread();
    }

    static JNIEnv *get_jni_env(void) {
        JNIEnv *env = (JNIEnv *) pthread_getspecific(current_jni_env);
        if (env == NULL) {
            env = attach_current_thread();
            pthread_setspecific(current_jni_env, env);
        }
        return env;
    }

    static void set_ui_message(const gchar *message, CustomData *data) {
        JNIEnv *env = get_jni_env();
        GST_DEBUG("Setting message to: %s", message);
        jstring jmessage = env->NewStringUTF(message);
        env->CallVoidMethod(data->app, set_message_method_id, jmessage);
        if (env->ExceptionCheck()) {
            GST_ERROR("Failed to call Java method");
            env->ExceptionClear();
        }
        env->DeleteLocalRef(jmessage);
    }
    static void
    error_cb(GstBus *bus, GstMessage *msg, CustomData *data) {
        GError *err;
        gchar *debug_info;
        gchar *message_string;

        gst_message_parse_error(msg, &err, &debug_info);
        message_string =
                g_strdup_printf("Error received from element %s: %s",
                                GST_OBJECT_NAME (msg->src), err->message);
        g_clear_error(&err);
        g_free(debug_info);
        set_ui_message(message_string, data);
        g_free(message_string);
        gst_element_set_state(data->pipeline, GST_STATE_NULL);
        gst_element_set_state(data->pipeline, GST_STATE_READY);
        gst_element_set_state(data->pipeline, GST_STATE_PLAYING);
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "error_cb");

    }
    static void
    state_changed_cb(GstBus *bus, GstMessage *msg, CustomData *data) {
        GstState old_state, new_state, pending_state;
        gst_message_parse_state_changed(msg, &old_state, &new_state, &pending_state);
        if (GST_MESSAGE_SRC (msg) == GST_OBJECT (data->pipeline)) {
            gchar *message = g_strdup_printf("State changed to %s",
                                             gst_element_state_get_name(new_state));
            set_ui_message(message, data);
            g_free(message);
        }
    }

    static void check_initialization_complete(CustomData *data) {
        JNIEnv *env = get_jni_env();
        if (!data->initialized && data->native_window && data->main_loop) {
            GST_DEBUG
            ("Initialization complete, notifying application. native_window:%p main_loop:%p",
             data->native_window, data->main_loop);

            gst_video_overlay_set_window_handle(GST_VIDEO_OVERLAY(data->video_sink),
                                                (guintptr) data->native_window);
            env->CallVoidMethod(data->app, on_gstreamer_initialized_method_id);
            if (env->ExceptionCheck()) {
                GST_ERROR("Failed to call Java method");
                env->ExceptionClear();
            }
            data->initialized = TRUE;
        }
    }
    
    bool check_video_restart(CustomData *data, GstBuffer *buffer) {
        bool restart_detected = false;
        const int MAX_FRAME_GAP_MS = 500;  // Maximum allowed gap between frames in milliseconds
        const gint64 MAX_PTS_GAP = GST_SECOND / 4;  // 250ms maximum PTS gap
        
        GstClockTime pts = GST_BUFFER_PTS(buffer);
        auto current_time = std::chrono::steady_clock::now();
        
    if (data->first_frame) {
        data->last_pts = pts;
        data->last_frame_time = current_time;
        data->first_frame = false;
        data->frame_count = 1;
        return false;
    }
    
    gint64 pts_diff = GST_CLOCK_DIFF(data->last_pts, pts);
    bool pts_gap = (pts_diff > MAX_PTS_GAP || pts_diff < 0);
    
    auto frame_time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - data->last_frame_time).count();
    bool timing_gap = (frame_time_diff > MAX_FRAME_GAP_MS);
    
    restart_detected = pts_gap || timing_gap;
    
    if (restart_detected) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, 
            "Video restart detected: pts_gap=%s, timing_gap=%s, gap_duration=%lldms",
            pts_gap ? "true" : "false",
            timing_gap ? "true" : "false",
            (long long)frame_time_diff);
            nativeflag=false;

        // Reset stabilization state
        prevOrig1 = cv::Mat();
        prevFrame1 = cv::Mat();
        prevGray1 = cv::Mat();
        count1 = 0;
    }
    
    data->last_pts = pts;
    data->last_frame_time = current_time;
    data->frame_count++;
    
    return restart_detected;
}

void ProcessFrame(cv::Mat &frame, ANativeWindow *native_window ,int width, int height) {
    if (!native_window) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Native window is not available.");
        nativeflag=false;
        return;
    }
        try {
    ANativeWindow_setBuffersGeometry(native_window, width, height, WINDOW_FORMAT_RGBX_8888);
    cv::Mat resized_frame;
    if (frame.cols != width || frame.rows != height) {
        cv::resize(frame, resized_frame, cv::Size(width, height));
    } else {
        resized_frame = frame;
    }
        
    ANativeWindow_Buffer buffer;
    if (ANativeWindow_lock(native_window, &buffer, nullptr) < 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to lock the native window.");
        return;
    }
    
    uint8_t *dst   = static_cast<uint8_t *>(buffer.bits);
    int dst_stride = buffer.stride * 4;  // 4 bytes per pixel (RGBA)
    int src_stride = resized_frame.step[0];  // Step of the resized frame

    for (int y = 0; y < height; ++y) {
        memcpy(dst + y * dst_stride, resized_frame.ptr(y), src_stride);
    }

    ANativeWindow_unlockAndPost(native_window);

    } catch (const cv::Exception &e) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "OpenCV exception in ProcessFrame: %s", e.what());
    } catch (const std::exception &e) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Standard exception in ProcessFrame: %s", e.what());
    } catch (...) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Unknown exception in ProcessFrame.");
    }
}

int getCurrentCore() {
    return sched_getcpu();
}
std::vector<cv::Mat> roi_gray(cv::Mat frame,int width,int height){
    cv::Mat prevGray, prevFrame;
    cv::Point top_left(height / roiDiv, width / roiDiv);
    cv::Point bottom_right(height - (height / roiDiv), width - (width / roiDiv));
    cv::Size frameSize(width, height);

    if (downSample != 1.0){
        resize(frame, frame, frameSize);
    }
    prevFrame = frame.clone();
    cv::Rect roi = cv::Rect(top_left.y, top_left.x, bottom_right.y-top_left.y, bottom_right.x-top_left.x);
    cvtColor(frame, prevGray, cv::COLOR_BGR2GRAY);
    prevGray = prevGray(roi);
    std::vector<cv::Mat> out;
    out.push_back(prevFrame);
    out.push_back(prevGray);
    return out;
}

cv::Mat func(int var, bool ret, cv::Mat frame, cv::Mat prevOrig, cv::Mat prevFrame, cv::Mat prevGray, cv::Mat currFrame, cv::Mat currGray, gpointer user_data, int width, int height) {
    CustomData *data = (CustomData*) user_data;
    cv::Mat f_stabilized;  // Declare outside try block
    cv::Mat Orig;          // Declare outside try block
    
    try {
        double res_w_orig = width;
        double res_h_orig = height;
        double res_w = res_w_orig * downSample;
        double res_h = res_h_orig * downSample;
        cv::Mat T = getRotationMatrix2D(cv::Point2f(res_w_orig / 2, res_h_orig / 2), 0, zoomFactor);

        if(!ret) {
            throw std::runtime_error("Frame not found");
        }
        Orig = frame.clone();
        f_stabilized = Orig.clone();  // Initialize with original frame
        
        if (prevFrame.empty()) {
            prevOrig = frame.clone();
            prevFrame = frame.clone();
            prevGray = currGray.clone();
            // Return early to avoid processing empty frames
            return f_stabilized;
        }

        std::vector<cv::Point2f> prevPts, currPts;
        goodFeaturesToTrack(prevGray, prevPts, 400, 0.01, 30, cv::Mat(), 3, false, 0.04);
        
        if(prevPts.empty()) {
            throw std::logic_error("prevpts found empty");
        }
        
        std::vector<uchar> status;
        std::vector<float> err;
        calcOpticalFlowPyrLK(prevGray, currGray, prevPts, currPts, status, err, winSize, 3, termcrit, 0, 0.001);
        
        if(currPts.empty()) {
            throw std::logic_error("currpts found empty");
        }

        std::vector<cv::Point2f> prevPts_rescaled, currPts_rescaled;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i] == 1) {
                prevPts_rescaled.push_back(prevPts[i] + cv::Point2f(res_w_orig / roiDiv, res_h_orig / roiDiv));
                currPts_rescaled.push_back(currPts[i] + cv::Point2f(res_w_orig / roiDiv, res_h_orig / roiDiv));
            }
        }

        if (prevPts_rescaled.size() < 3 || currPts_rescaled.size() < 3) {
            if(count1 == 0) {
                std::unique_lock<std::mutex> lock(mtx);
                switch(var) {
                    case 1: cv1.wait(lock, []{ return ready1; }); break;
                    case 2: cv1.wait(lock, []{ return ready2; }); break;
                    case 3: cv1.wait(lock, []{ return ready3; }); break;
                    default: cv1.wait(lock, []{ return ready4; }); break;
                }
                X_estimate = cv::Mat::zeros(1, 3, CV_64F);
                P_estimate = cv::Mat::ones(1, 3, CV_64F);
            }
            throw std::logic_error("insufficient points");
        }

        cv::Mat m = estimateAffinePartial2D(prevPts_rescaled, currPts_rescaled);
        if (m.empty()) {
            m = lastRigidTransform.clone();  // Use clone() for safety
        }

        double dx = m.at<double>(0, 2);
        double dy = m.at<double>(1, 2);
        double da = atan2(m.at<double>(1, 0), m.at<double>(0, 0));
        
        y += dy;
        a += da;
        
        cv::Mat Z = (cv::Mat_<double>(1, 3) << x, y, a);
        
        if (count1 == 0) {
            X_estimate = cv::Mat::zeros(1, 3, CV_64F);
            P_estimate = cv::Mat::ones(1, 3, CV_64F);
        } else {
            cv::Mat X_predict = X_estimate.clone();
            cv::Mat P_predict = P_estimate + Q;
            cv::Mat K = P_predict / (P_predict + R);
            X_estimate = X_predict + K.mul(Z - X_predict);
            P_estimate = (cv::Mat::ones(1, 3, CV_64F) - K).mul(P_predict);
        }

        double diff_x = X_estimate.at<double>(0, 0) - x;
        double diff_y = X_estimate.at<double>(0, 1) - y;
        double diff_a = X_estimate.at<double>(0, 2) - a;
        
        dx += diff_x;
        dy += diff_y;
        da += diff_a;

        cv::Mat m_new = (cv::Mat_<double>(2, 3) << cos(da), -sin(da), dx,
                                                   sin(da), cos(da), dy);
        
        cv::Mat fS;  // Temporary matrix for intermediate result
        warpAffine(prevOrig, fS, m_new, cv::Size(res_w_orig, res_h_orig));
        warpAffine(fS, f_stabilized, T, cv::Size(fS.cols, fS.rows));

        lastRigidTransform = m_new.clone();
        count1++;
        
    } catch(const std::runtime_error& e1) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Exception in catch ---1: %s", e1.what());
        f_stabilized = Orig.clone();
        count1++;
    } catch (const std::logic_error& e2) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Exception in catch ---2: %s", e2.what());
        f_stabilized = Orig.clone();
        count1++;
    } catch(std::exception& e) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Exception in catch ---3: %s", e.what());
        f_stabilized = Orig.clone();
        count1++;
    }
    if (var == 1) {
        ready1 = false;
        ready2 = true;
    } else if (var == 2) {
        ready2 = false;
        ready3 = true;
    } else if (var == 3) {
        ready3 = false;
        ready4 = true;
    } else {
        ready4 = false;
        ready1 = true;
    }

    cv1.notify_all();
    return f_stabilized;
}

int output(gpointer user_data){
    CustomData *data = (CustomData*) user_data;  // Cast user_data to CustomData*

    while(true){
    try {

        if (nativeflag.load() == true) {

        std::future<cv::Mat> fut;
        mtx2.lock();

        int future_size = futures.size();
        if (future_size>0){
            fut = std::move(futures[0]);
            futures.erase(futures.begin());
            mtx2.unlock();
        }
        else{
            mtx2.unlock();
            continue;
        }
            cv::Mat out = fut.get();
            ProcessFrame(out, data->native_window,data->incomingwidth, data->incomingheight);
            // __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Flag is set to true... output");
            }
        } catch (const std::exception &e) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Exception in output function: %s", e.what());
        } catch (...) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Unknown exception in output function");
        }
    }
    return 0;
}

GstFlowReturn on_new_sample(GstElement *sink, gpointer user_data) {
    CustomData *data = (CustomData*) user_data;  // Cast user_data to CustomData*
    GstSample *sample;
    GstBuffer *buffer;
    GstMapInfo map;
    GstCaps *caps;
    GstStructure *structure;
    int incomingwidth, incomingheight;
    try {
    sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to pull sample.");
        return GST_FLOW_ERROR;
    }
    buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to get b-uffer from sample.");
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }
    caps = gst_sample_get_caps(sample);
    if (!caps) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to get caps from sample.");
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }
    // if (check_video_restart(data, buffer)) {
    //     nativeflag = false;  // Reset processing if needed
    //     __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Video feed restart detected, resetting processing state");
    // }

    // Get the structure from caps
    structure = gst_caps_get_structure(caps, 0);
    if (!gst_structure_get_int(structure, "width", &incomingwidth) ||
        !gst_structure_get_int(structure, "height", &incomingheight)) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to get width/height from caps.");
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }
    if (incomingwidth != data->prev_width || incomingheight != data->prev_height) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Resolution change detected. Resetting stabilization.");

        // Reset the stabilization variables
        prevOrig1 = cv::Mat();
        prevFrame1 = cv::Mat();
        prevGray1 = cv::Mat();

        // Update previous dimensions in CustomData
        data->prev_width = incomingwidth;
        data->prev_height = incomingheight;
    }
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to map buffer.");
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }
    data->incomingwidth = incomingwidth;
    data->incomingheight = incomingheight;

    int width = incomingwidth;
    int height = incomingheight;
    cv::Mat frame(height, width, CV_8UC3, (void*)map.data, cv::Mat::AUTO_STEP);
    cv::cvtColor(frame, frame, cv::COLOR_RGB2RGBA);
    if (nativeflag.load() == false || incomingwidth==684 && incomingheight==512 || incomingwidth==512 && incomingheight==384 || incomingwidth==160 && incomingheight==120 ) {
        ProcessFrame(frame, data->native_window,width,height);
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
    if (nativeflag.load() == true) {

        bool ret = !frame.empty();
        if (ret == 0){
            currOrig1 = cv::Mat();
            currFrame1 = cv::Mat();
            currGray1 = cv::Mat();
        }
        else{
            currOrig1 = frame.clone();
            std::vector<cv::Mat> out1 = roi_gray(frame,width,height);
            currFrame1 = out1[0];
            currGray1 = out1[1];
        }
        mtx2.lock();

            if (num_cores_processing == -1){
                futures.push_back(std::async(std::launch::async, func, (num) % 4,ret,frame,prevOrig1,prevFrame1,prevGray1,currFrame1,currGray1,data,width,height));
            }
        else{
            while(true){
                if (futures.size()<num_cores_processing) {
                    futures.push_back(
                            std::async(std::launch::async, func, (num) % 4, ret, frame, prevOrig1,
                                    prevFrame1, prevGray1, currFrame1, currGray1, data,width,height));
                    break;
                }
            }
        }
        
        mtx2.unlock();
        prevOrig1 = currOrig1.clone();
        prevGray1 = currGray1.clone();
        prevFrame1 = currFrame1.clone();
        num = num+1;
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);

            return GST_FLOW_OK;
        }
    } catch (const std::exception &e) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Exception in on_new_sample: %s", e.what());
        if (sample) gst_sample_unref(sample);
        if (buffer) gst_buffer_unmap(buffer, &map);
        return GST_FLOW_ERROR;
    } catch (...) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Unknown exception in on_new_sample");
        if (sample) gst_sample_unref(sample);
        if (buffer) gst_buffer_unmap(buffer, &map);
        return GST_FLOW_ERROR;
    }

    return GST_FLOW_OK;
}

    static void *app_function(void *userdata) {
        JavaVMAttachArgs args;
        GstBus *bus;
        CustomData *data = (CustomData *) userdata;
        GSource *bus_source;
        GError *error = NULL;

        GST_DEBUG ("Creating pipeline in CustomData at %p", data);
        data->context = g_main_context_new();
        g_main_context_push_thread_default(data->context);
        std::future<int> t = std::async(std::launch::async, output,data);
        data->pipeline = gst_parse_launch("udpsrc port=5600 caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96\" ! queue ! rtph264depay ! avdec_h264 ! videoconvert ! videorate ! video/x-raw,format=RGB ! appsink name=appsink", &error);
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "pipeline: %s",data->pipeline);

        if (error) {
            gchar *message = g_strdup_printf("Unable to build pipeline: %s", error->message);
            g_clear_error(&error);
            set_ui_message(message, data);      
            g_free(message);
            return NULL;
        }
        GstElement *appsink = gst_bin_get_by_name(GST_BIN(data->pipeline), "appsink");
        if (!appsink) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "appsink element not found in pipeline");
            return NULL;
        }
        if (!GST_IS_APP_SINK(appsink)) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Retrieved element is  an appsink");
            gst_object_unref(appsink);
            return NULL;
        }
        gst_app_sink_set_emit_signals((GstAppSink*)appsink, true);
        gst_app_sink_set_drop((GstAppSink*)appsink, true);
        gst_app_sink_set_max_buffers((GstAppSink*)appsink, 1);
        g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), data);

        gst_element_set_state(data->pipeline, GST_STATE_READY);
        gst_element_set_state(data->pipeline, GST_STATE_PLAYING);
        gst_object_unref(data->video_sink);
        gst_object_unref(data->pipeline);
        return NULL;
    }

static void gst_native_init(JNIEnv *env, jobject thiz) {
    CustomData *data = g_new0(CustomData, 1);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "working in cpp");

    if (!data) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Failed to allocate memory for CustomData");
        __android_log_print(ANDROID_LOG_ERROR, "tutorial-3", "Failed to allocate memory for CustomData");
        return;
    }
    SET_CUSTOM_DATA(env, thiz, custom_data_field_id, data);

    GST_DEBUG_CATEGORY_INIT(debug_category, "tutorial-3", 0, "Android tutorial 3");
    gst_debug_set_threshold_for_name("tutorial-3", GST_LEVEL_DEBUG);
    GST_DEBUG("Created CustomData at %p", data);
    data->app = env->NewGlobalRef(thiz);
    GST_DEBUG("Created GlobalRef for app object at %p", data->app);
    pthread_create(&gst_app_thread, NULL, &app_function, data);

    if (env->ExceptionCheck()) {
        __android_log_print(ANDROID_LOG_ERROR, "tutorial-3", "Failed to set custom data");
        env->ExceptionClear();
        g_free(data);
        return;
    }
}
    static void gst_native_finalize(JNIEnv *env, jobject thiz) {
        CustomData *data = GET_CUSTOM_DATA(env, thiz, custom_data_field_id);
        if (!data)
            return;
        GST_DEBUG("Quitting main loop...");
        g_main_loop_quit(data->main_loop);
        GST_DEBUG("Waiting for thread to finish...");
        pthread_join(gst_app_thread, NULL);
        GST_DEBUG("Deleting GlobalRef for app object at %p", data->app);
        env->DeleteGlobalRef(data->app);
        GST_DEBUG("Freeing CustomData at %p", data);
        g_free(data);
        SET_CUSTOM_DATA(env, thiz, custom_data_field_id, NULL);
        GST_DEBUG("Done finalizing");
    }

    static void
    gst_native_play(JNIEnv *env, jobject thiz) {
        CustomData *data = GET_CUSTOM_DATA (env, thiz, custom_data_field_id);
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Setting state to PLAYING...", data);
        if (!data)
            return;
        GST_DEBUG ("Setting state to PLAYING");
        gst_element_set_state(data->pipeline, GST_STATE_PLAYING);
    }
    static void
    gst_native_pause(JNIEnv *env, jobject thiz) {
        CustomData *data = GET_CUSTOM_DATA (env, thiz, custom_data_field_id);
        if (!data)
            return;
        GST_DEBUG ("Setting state to PAUSED");
        gst_element_set_state(data->pipeline, GST_STATE_PAUSED);
    }

static jboolean gst_native_class_init(JNIEnv *env, jclass klass) {
    custom_data_field_id = env->GetFieldID(klass, "native_custom_data", "J");
    if (!custom_data_field_id) {
        __android_log_print(ANDROID_LOG_ERROR, "tutorial-3", "Failed to get field ID for native_custom_data");
        return JNI_FALSE;
    }

    set_message_method_id = env->GetMethodID(klass, "setMessage", "(Ljava/lang/String;)V");
    if (!set_message_method_id) {
        __android_log_print(ANDROID_LOG_ERROR, "tutorial-3", "Failed to get method ID for setMessage");
        return JNI_FALSE;
    }

    on_gstreamer_initialized_method_id = env->GetMethodID(klass, "onGStreamerInitialized", "()V");
    if (!on_gstreamer_initialized_method_id) {
        __android_log_print(ANDROID_LOG_ERROR, "tutorial-3", "Failed to get method ID for onGStreamerInitialized");
        return JNI_FALSE;
    }

    return JNI_TRUE;
}
    static void
    gst_native_surface_init(JNIEnv *env, jobject thiz, jobject surface) {
        CustomData *data = GET_CUSTOM_DATA (env, thiz, custom_data_field_id);
        if (!data)
            return;
        ANativeWindow *new_native_window = ANativeWindow_fromSurface(env, surface);
        GST_DEBUG ("Received surface %p (native window %p)", surface,
                   new_native_window);

        if (data->native_window) {
            ANativeWindow_release(data->native_window);
            if (data->native_window == new_native_window) {
                GST_DEBUG ("New native window is the same as the previous one %p",
                           data->native_window);
                if (data->video_sink) {
                    gst_video_overlay_expose(GST_VIDEO_OVERLAY (data->video_sink));
                    gst_video_overlay_expose(GST_VIDEO_OVERLAY (data->video_sink));
                }
                return;
            } else {
                GST_DEBUG ("Released previous native window %p", data->native_window);
                data->initialized = FALSE;
            }
        }
        data->native_window = new_native_window;

        check_initialization_complete(data);
    }

    static jstring
    gst_native_get_gstreamer_info(JNIEnv *env, jobject thiz) {
        char *version_utf8 = gst_version_string();
        jstring version_jstring = env->NewStringUTF(version_utf8);
        g_free(version_utf8);

        return version_jstring;
    }
    static void
    gst_native_surface_finalize(JNIEnv *env, jobject thiz) {
        CustomData *data = GET_CUSTOM_DATA (env, thiz, custom_data_field_id);
        if (!data)
            return;
        GST_DEBUG ("Releasing Native Window %p", data->native_window);

        if (data->video_sink) {
            gst_video_overlay_set_window_handle(GST_VIDEO_OVERLAY (data->video_sink),
                                                (guintptr) NULL);
            gst_element_set_state(data->pipeline, GST_STATE_READY);
        }

        ANativeWindow_release(data->native_window);
        data->native_window = NULL;
        data->initialized = FALSE;
    }
    static JNINativeMethod native_methods[] = {
        {"nativeGetGStreamerInfo", "()Ljava/lang/String;", (void *) gst_native_get_gstreamer_info},
        {"nativeInit", "()V", (void *) gst_native_init},
        {"nativeFinalize", "()V", (void *) gst_native_finalize},
        {"nativePlay",             "()V",                  (void *) gst_native_play},
        {"nativePause", "()V", (void *) gst_native_pause},
        {"nativeSurfaceInit", "(Ljava/lang/Object;)V",(void *) gst_native_surface_init},
        {"nativeSurfaceFinalize", "()V", (void *) gst_native_surface_finalize},
        {"nativeClassInit", "()Z", (void *) gst_native_class_init}
    };
    extern "C"
    JNIEXPORT void JNICALL
    Java_com_kalyzee_gstreamer_GstPlayer_nativeSetFlag(JNIEnv *env, jobject thiz, jboolean flag) {
        nativeflag.store((bool) flag);
        if (nativeflag.load() ) {
            __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Flag is set to true...");

        } else {
            __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Flag is set to false...");
        }
    }

jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    JNIEnv *env = NULL;
    java_vm = vm;  // Add this line to set the global java_vm variable

    if (vm->GetEnv((void **) &env, JNI_VERSION_1_4) != JNI_OK) {
        __android_log_print(ANDROID_LOG_ERROR, "native-lib", "Could not retrieve JNIEnv");
        return 0;
    }
    jclass klass = env->FindClass("com/kalyzee/gstreamer/GstPlayer");
    if (klass == NULL) {
        __android_log_print(ANDROID_LOG_ERROR, "native-lib", "Could not find Tutorial3 class");
        return 0;
    }
    if (env->RegisterNatives(klass, native_methods, G_N_ELEMENTS(native_methods)) < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "native-lib", "Could not register native methods");
        return 0;
    }

    return JNI_VERSION_1_4;
}