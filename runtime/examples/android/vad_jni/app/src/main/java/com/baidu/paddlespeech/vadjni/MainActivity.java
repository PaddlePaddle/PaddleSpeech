package com.baidu.paddlespeech.vadjni;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;

import com.baidu.paddlespeech.vadjni.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'vadjni' library on application startup.
    static {
        System.loadLibrary("vadjni");
    }

    private ActivityMainBinding binding;
    private long instance;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Example of a call to a native method
        TextView tv = binding.sampleText;
        tv.setText(stringFromJNI());

        Button lw = binding.loadWav;

        instance = createInstance();
    }

    /**
     * A native method that is implemented by the 'vadjni' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    public native long createInstance(String config_path);

    public native int destroyInstance(long instance);

    public native int reset(long instance);

    public native int chunkSizeSamples(long instance);

    public native int feedForward(long instance, float[] chunk);
}