//******************* NOT ABLE TO SEE THE VIDEO OUTPUT WINDOW IN REACT NATIVE  **********************

// package com.kalyzee.gstreamer;

// import android.app.Activity;
// import android.os.Bundle;
// // import android.support.annotation.Nullable;

// // ***The above import statement is deprecated so below we are using new one ****
// import androidx.annotation.Nullable;

// import android.util.Log;
// import android.widget.ProgressBar;

// import com.facebook.react.common.MapBuilder;
// import com.kalyzee.gstreamer.GstPlayer;

// import android.widget.Toast;

// import com.facebook.drawee.backends.pipeline.Fresco;
// import com.facebook.react.bridge.ReactApplicationContext;
// import com.facebook.react.bridge.ReactContext;
// import com.facebook.react.uimanager.SimpleViewManager;
// import com.facebook.react.uimanager.ThemedReactContext;
// import com.facebook.react.uimanager.ViewManager;
// import com.facebook.react.uimanager.ViewProps;
// import com.facebook.react.uimanager.annotations.ReactProp;
// import com.facebook.react.views.image.ImageResizeMode;
// import com.facebook.react.views.image.ReactImageManager;
// import com.facebook.react.views.image.ReactImageView;

// import java.util.Arrays;
// import java.util.List;
// import java.util.Map;


// public class ReactGstPlayerManager extends SimpleViewManager<GstPlayer> {

//     public static final String REACT_CLASS = "GstPlayer";

//     @Override
//     public String getName() {
//         return REACT_CLASS;
//     }

//     @Override
//     protected GstPlayer createViewInstance(ThemedReactContext reactContext) {
//         GstPlayer gst = new GstPlayer(reactContext);
//         return gst;
//     }

//     @ReactProp(name = "uri")
//     public void setUri(GstPlayer view, String uri) {
//         view.setMediaUri(uri);
//     }

//     @ReactProp(name = "play")
//     public void setPlayState(GstPlayer view, boolean play) {
//         view.setPlay(play);
//     }


//     /**
//      * This method maps the sending of the "onAudioLevelChange" event to the JS "onAudioLevelChange" function.
//      */
//     @Nullable @Override
//     public Map<String, Object> getExportedCustomDirectEventTypeConstants() {
//         return MapBuilder.<String, Object>builder()
//                 .put("onAudioLevelChange", MapBuilder.of(
//                         "registrationName", "onAudioLevelChange")
//                 ).build();
//     }
// }




package com.kalyzee.gstreamer;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.uimanager.SimpleViewManager;
import com.facebook.react.uimanager.ThemedReactContext;
import com.facebook.react.uimanager.annotations.ReactProp;
import com.facebook.react.common.MapBuilder;

import java.util.Map;

import androidx.annotation.Nullable;

public class ReactGstPlayerManager extends SimpleViewManager<GstPlayer> {

    public static final String REACT_CLASS = "GstPlayer";

    @Override
    public String getName() {
        return REACT_CLASS;
    }

    @Override
    protected GstPlayer createViewInstance(ThemedReactContext reactContext) {
        return new GstPlayer(reactContext);
    }

    // @ReactProp(name = "uri")
    // public void setUri(GstPlayer view, String uri) {
    //     // view.setMediaUri(uri);
    // }

    @ReactProp(name = "play")
    public void setPlayState(GstPlayer view, boolean play) {
        view.setPlay(play);
    }

    
    @Override
    public Map<String, Object> getExportedCustomDirectEventTypeConstants() {
        return MapBuilder.<String, Object>builder()
                .put("onAudioLevelChange", MapBuilder.of(
                        "registrationName", "onAudioLevelChange")
                ).build();
    }

    @ReactProp(name = "nativeFlag")
    public void setNativeFlag(GstPlayer view, boolean flag) {
        view.setNativeFlag(flag); // You'll implement this method in your GstPlayer class
}

}