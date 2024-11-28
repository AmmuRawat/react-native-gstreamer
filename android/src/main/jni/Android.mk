LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# OpenCV configuration
OPENCV_INSTALL_MODULES := on
OPENCV_LIB_TYPE := SHARED
OpenCV_DIR=/home/botlab/AndroidStudioProjects/opencv-4.10.0-android-sdk/OpenCV-android-sdk/sdk/native

include ${OpenCV_DIR}/jni/OpenCV.mk

# GStreamer configuration
GSTREAMER_ROOT_ANDROID=/home/botlab/Kunal_meena/react_native/github_repo_usb_android/gstreamer_project/gstreamer-1.0-android-universal-1.20.6

ifndef GSTREAMER_ROOT_ANDROID
    $(error GSTREAMER_ROOT_ANDROID is not defined!)
endif

ifeq ($(TARGET_ARCH_ABI),armeabi)
    GSTREAMER_ROOT := $(GSTREAMER_ROOT_ANDROID)/arm
else ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
    GSTREAMER_ROOT := $(GSTREAMER_ROOT_ANDROID)/armv7
else ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
    GSTREAMER_ROOT := $(GSTREAMER_ROOT_ANDROID)/arm64
else ifeq ($(TARGET_ARCH_ABI),x86)
    GSTREAMER_ROOT := $(GSTREAMER_ROOT_ANDROID)/x86
else ifeq ($(TARGET_ARCH_ABI),x86_64)
    GSTREAMER_ROOT := $(GSTREAMER_ROOT_ANDROID)/x86_64
else
    $(error Target arch ABI not supported: $(TARGET_ARCH_ABI))
endif

GSTREAMER_NDK_BUILD_PATH := $(GSTREAMER_ROOT)/share/gst-android/ndk-build/

include $(GSTREAMER_NDK_BUILD_PATH)/plugins.mk

GSTREAMER_PLUGINS         :=  $(GSTREAMER_PLUGINS_CORE)      \
							  $(GSTREAMER_PLUGINS_PLAYBACK)  \
							  $(GSTREAMER_PLUGINS_CODECS)    \
							  $(GSTREAMER_PLUGINS_NET)       \
							  $(GSTREAMER_PLUGINS_SYS)       \
							  $(GSTREAMER_PLUGINS_CODECS_RESTRICTED) \
							  $(GSTREAMER_CODECS_GPL)        \
							  $(GSTREAMER_PLUGINS_ENCODING)  \
							  $(GSTREAMER_PLUGINS_VIS)       \
							  $(GSTREAMER_PLUGINS_EFFECTS)   \
							  $(GSTREAMER_PLUGINS_NET_RESTRICTED) \
#                              $(GSTREAMER_PLUGINS_CAPTURE)

GSTREAMER_PLUGINS         += $(GSTREAMER_PLUGINS_APP)

G_IO_MODULES := openssl

GSTREAMER_EXTRA_DEPS := gstreamer-1.0 gstreamer-video-1.0 glib-2.0 gstreamer-rtsp-server-1.0 gstreamer-app-1.0
LOCAL_MODULE := native-lib

LOCAL_SRC_FILES := native-lib.cpp

LOCAL_C_INCLUDES += $(OPENCV_DIR)/sdk/native/jni/include
LOCAL_CFLAGS += -frtti -fexceptions -fopenmp -w

LOCAL_SHARED_LIBRARIES := gstreamer_android  opencv_java4

LOCAL_LDLIBS := -llog -landroid -lc++

include $(BUILD_SHARED_LIBRARY)

# Include GStreamer make file
include $(GSTREAMER_NDK_BUILD_PATH)/gstreamer-1.0.mk

