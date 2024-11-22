//******************* NOT ABLE TO SEE THE VIDEO OUTPUT WINDOW IN REACT NATIVE  **********************

// package com.kalyzee.gstreamer;

// import android.content.Context;
// import android.os.Bundle;
// import android.util.AttributeSet;
// import android.util.Log;

// import android.view.View;
// import android.view.SurfaceView;
// import android.view.SurfaceHolder;

// import android.os.PowerManager;

// import com.facebook.react.bridge.Arguments;
// import com.facebook.react.bridge.ReactContext;
// import com.facebook.react.bridge.WritableMap;
// import com.facebook.react.uimanager.events.RCTEventEmitter;

// import org.freedesktop.gstreamer.GStreamer;

// public class GstPlayer extends SurfaceView implements SurfaceHolder.Callback {

//     private native void nativeInit();     // Initialize native code, build pipeline, etc

//     private native void nativeFinalize(); // Destroy pipeline and shutdown native code

//     private native void nativePlay();     // Set pipeline to PLAYING

//     private native void nativePause();    // Set pipeline to PAUSED

//     private native void nativeSetUri(String uri); // Set the URI of the media to play

//     private static native boolean nativeClassInit(); // Initialize native class: cache Method IDs for callbacks

//     private native void nativeSetPosition(int milliseconds); // Seek to the indicated position, in milliseconds

//     private native void nativeSurfaceInit(Object surface);

//     private native void nativeSurfaceFinalize();

//     private boolean isReady;

//     private int position;                 // Current position, reported by native code
//     private int duration;                 // Current clip duration, reported by native code
//     private boolean is_local_media;       // Whether this clip is stored locally or is being streamed
//     private int desired_position;         // Position where the users wants to seek to
//     private String mediaUri;              // URI of the clip being played
//     private final String defaultMediaUri = ""; //"http://ftp.nluug.nl/pub/graphics/blender/demo/movies/Sintel.2010.1080p.mkv";


//     static private final int PICK_FILE_CODE = 1;
//     private String last_folder;

//     //private PowerManager.WakeLock wake_lock;

//     private long native_custom_data;      // Native code will use this to keep private data
//     private boolean is_playing_desired = true;   // Whether the user asked to go to PLAYING


//     public GstPlayer(Context context, AttributeSet attrs,
//                      int defStyle) {
//         super(context, attrs, defStyle);
//         try {
//             GStreamer.init(getContext());
//         } catch (Exception e) {
//         }
//         mediaUri = defaultMediaUri;
//         SurfaceHolder sh = this.getHolder();
//         sh.addCallback(this);
//         nativeInit();
//     }

//     public GstPlayer(Context context, AttributeSet attrs) {
//         this(context, attrs, 0);
//     }

//     public GstPlayer(Context context) {
//         this(context, null);
//     }


//     public void surfaceChanged(SurfaceHolder holder, int format, int width,
//                                int height) {
//         Log.i("gst-player", "Surface changed to format " + format + " width "
//                 + width + " height " + height);
//         nativeSurfaceInit(holder.getSurface());
//     }

//     // Set the URI to play, and record whether it is a local or remote file
//     public void setMediaUri(String uri) {
//         this.mediaUri = uri;
//         nativeSetUri(mediaUri);
//     }

//     public void surfaceCreated(SurfaceHolder holder) {
//         Log.i("gst-player", "Surface created: " + holder.getSurface());
//     }

//     public void surfaceDestroyed(SurfaceHolder holder) {
//         Log.i("gst-player", "Surface destroyed");
//         nativeSurfaceFinalize();
//     }

//     private void setCurrentPosition(final int position, final int duration) {

//     }

//     private void onMediaSizeChanged(int width, int height) {

//     }

//     private void onAudioLevelChange(float audio_level) {
//         Log.i("gst-player", "Audio level : " + audio_level);

//         final Context context = getContext();
//         if (context instanceof ReactContext) {

//             WritableMap event = Arguments.createMap();
//             event.putDouble("level", audio_level);

//             ((ReactContext) context).getJSModule(RCTEventEmitter.class).receiveEvent(
//                     getId(), "onAudioLevelChange", event
//             );
//         }
//     }

//     // Called from native code. This sets the content of the TextView from the UI thread.
//     private void setMessage(final String message) {

//         if (message.equals("State changed to READY")) {
//             Log.i("gst-player", "READY " + this.mediaUri);
//             nativePlay();
//         }
//     }

//     protected void onDestroy() {
//         nativeFinalize();
//     }


//     // Called from native code. Native code calls this once it has created its pipeline and
//     // the main loop is running, so it is ready to accept commands.
//     private void onGStreamerInitialized() {
//         Log.i("gst-player", "Gst initialized. Restoring state, playing:" + is_playing_desired);
//         nativeSetUri(mediaUri);
//         nativeSetPosition(position);
//         if (is_playing_desired) {
//             nativePlay();
//         } else {
//             nativePause();
//         }
//     }


//     public void setPlay(boolean play) {

//         is_playing_desired = play;

//         if (play && isReady) {
//             nativePlay();
//         } else {
//             if (!play && isReady) {
//                 nativePause();
//             }
//         }
//     }

//     static {
//         System.loadLibrary("gstreamer_android");
//         System.loadLibrary("gstreamer");
//         nativeClassInit();
//     }
// }


package com.kalyzee.gstreamer;

import android.content.Context;
import android.graphics.SurfaceTexture;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Surface;
import android.view.TextureView;
import android.view.TextureView.SurfaceTextureListener;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReactContext;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.uimanager.events.RCTEventEmitter;

import org.freedesktop.gstreamer.GStreamer;

public class GstPlayer extends TextureView implements SurfaceTextureListener {

    private native String nativeGetGStreamerInfo();
    private native void nativePlay();
    private native void nativeInit();     // Initialize native code, build pipeline, etc
    private native void nativeFinalize(); // Destroy pipeline and shutdown native code
    private native void nativePause();    // Set pipeline to PAUSED
    private static native boolean nativeClassInit(); // Initialize native class: cache Method IDs for callbacks
    private native void nativeSurfaceInit(Object surface);
    private native void nativeSurfaceFinalize();
    private long native_custom_data;      // Native code will use this to keep private data
    private native void nativeSetFlag(boolean flag);

    // private boolean is_playing_desired;   // Whether the user asked to go to PLAYING

    private boolean isReady;

    private int position;                 // Current position, reported by native code
    private int duration;                 // Current clip duration, reported by native code
    private boolean is_local_media;       // Whether this clip is stored locally or is being streamed
    private int desired_position;         // Position where the users wants to seek to
    private String mediaUri;              // URI of the clip being played
    private final String defaultMediaUri = ""; //"http://ftp.nluug.nl/pub/graphics/blender/demo/movies/Sintel.2010.1080p.mkv";

    static private final int PICK_FILE_CODE = 1;
    private String last_folder;

    // private long native_custom_data;      // Native code will use this to keep private data
    private boolean is_playing_desired = true;   // Whether the user asked to go to PLAYING

    public GstPlayer(Context context, AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
        try {
            GStreamer.init(getContext());
        } catch (Exception e) {
            // Handle exception
        }
        mediaUri = defaultMediaUri;
        setSurfaceTextureListener(this);
        nativeInit();
    }

    public GstPlayer(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public GstPlayer(Context context) {
        this(context, null);
    }
    
    public void setFixedVideoSize(int width, int height) {
        getLayoutParams().width = width;
        getLayoutParams().height = height;
        requestLayout();
    }
    
    
    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
        Log.i("gst-player", "Surface created: " + surface);
        setFixedVideoSize(1280, 720);  // Set the fixed size here
        nativeSurfaceInit(new Surface(surface));
    }

    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
        Log.i("gst-player", "Surface changed to width " + width + " height " + height);
    }

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
        Log.i("gst-player", "Surface destroyed");
        nativeSurfaceFinalize();
        return true;
    }

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        // No action needed here
    }

    private void setCurrentPosition(final int position, final int duration) {
        // Handle setting current position
    }

    private void onMediaSizeChanged(int width, int height) {
        // Handle media size change
    }

    private void onAudioLevelChange(float audio_level) {
        Log.i("gst-player", "Audio level: " + audio_level);

        final Context context = getContext();
        if (context instanceof ReactContext) {
            WritableMap event = Arguments.createMap();
            event.putDouble("level", audio_level);

            ((ReactContext) context).getJSModule(RCTEventEmitter.class).receiveEvent(
                    getId(), "onAudioLevelChange", event
            );
        }
    }

    private void setMessage(final String message) {
        if (message.equals("State changed to READY")) {
            Log.i("gst-player", "READY " + this.mediaUri);
            // nativePlay();
        }
    }

    protected void onDestroy() {
        nativeFinalize();
    }

    private void onGStreamerInitialized() {
        Log.i("gst-player", "Gst initialized. Restoring state, playing:" + is_playing_desired);
        // nativeSetPosition(position);
        if (is_playing_desired) {
            // nativePlay();
        } else {
            nativePause();
        }
    }

    public void setPlay(boolean play) {
        is_playing_desired = play;
        if (play && isReady) {
            // nativePlay();
        } else {
            if (!play && isReady) {
                nativePause();
            }
        }
    }

    public void setNativeFlag(boolean flag) {
        nativeSetFlag(flag);  // This calls the C++ method
    }


    // public void setMediaUri(String uri) {
    //     this.mediaUri = uri;
    //     // nativeSetUri(uri);
    // }

    static {
        System.loadLibrary("gstreamer_android");
//        System.loadLibrary("tutorial-3");
        System.loadLibrary("native-lib");
        // System.loadLibrary("opencv_java4");

//        System.loadLibrary("wireless-streaming");
        nativeClassInit();
    }
}