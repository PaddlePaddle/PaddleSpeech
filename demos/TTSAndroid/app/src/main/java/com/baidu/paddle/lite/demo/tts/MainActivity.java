package com.baidu.paddle.lite.demo.tts;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.preference.PreferenceManager;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;

public class MainActivity extends AppCompatActivity implements View.OnClickListener, MediaPlayer.OnPreparedListener, MediaPlayer.OnErrorListener, AdapterView.OnItemSelectedListener {
    public static final int REQUEST_LOAD_MODEL = 0;
    public static final int REQUEST_RUN_MODEL = 1;
    public static final int RESPONSE_LOAD_MODEL_SUCCESSED = 0;
    public static final int RESPONSE_LOAD_MODEL_FAILED = 1;
    public static final int RESPONSE_RUN_MODEL_SUCCESSED = 2;
    public static final int RESPONSE_RUN_MODEL_FAILED = 3;
    public MediaPlayer mediaPlayer = new MediaPlayer();
    private static final String TAG = Predictor.class.getSimpleName();
    protected ProgressDialog pbLoadModel = null;
    protected ProgressDialog pbRunModel = null;
    // Receive messages from worker thread
    protected Handler receiver = null;
    // Send command to worker thread
    protected Handler sender = null;
    // Worker thread to load&run model
    protected HandlerThread worker = null;
    // UI components of image classification
    protected TextView tvInputSetting;
    protected TextView tvInferenceTime;
    protected Button btn_play;
    protected Button btn_pause;
    protected Button btn_stop;
    // Model settings of image classification
    protected String modelPath = "";
    protected int cpuThreadNum = 1;
    protected String cpuPowerMode = "";
    protected Predictor predictor = new Predictor();
    int sampleRate = 24000;
    private final String wavName = "tts_output.wav";
    private final String wavFile = Environment.getExternalStorageDirectory() + File.separator + wavName;
    private final String AMmodelName = "fastspeech2_csmsc_arm.nb";
    private final String VOCmodelName = "mb_melgan_csmsc_arm.nb";
    private float[] phones = {};
    private final float[][] sentencesToChoose = {
            // 009901 昨日，这名“伤者”与医生全部被警方依法刑事拘留。
            {261, 231, 175, 116, 179, 262, 44, 154, 126, 177, 19, 262, 42, 241, 72, 177, 56, 174, 245, 37, 186, 37, 49, 151, 127, 69, 19, 179, 72, 69, 4, 260, 126, 177, 116, 151, 239, 153, 141},
            // 009902 钱伟长想到上海来办学校是经过深思熟虑的。
            {174, 83, 213, 39, 20, 260, 89, 40, 30, 177, 22, 71, 9, 153, 8, 37, 17, 260, 251, 260, 99, 179, 177, 116, 151, 125, 70, 233, 177, 51, 176, 108, 177, 184, 153, 242, 40, 45},
            // 009903 她见我一进门就骂，吃饭时也骂，骂得我抬不起头。
            {182, 2, 151, 85, 232, 73, 151, 123, 154, 52, 151, 143, 154, 5, 179, 39, 113, 69, 17, 177, 114, 105, 154, 5, 179, 154, 5, 40, 45, 232, 182, 8, 37, 186, 174, 74, 182, 168},
            // 009904 李述德在离开之前，只说了一句“柱驼杀父亲了”。
            {153, 74, 177, 186, 40, 42, 261, 10, 153, 73, 152, 7, 262, 113, 174, 83, 179, 262, 115, 177, 230, 153, 45, 73, 151, 242, 180, 262, 186, 182, 231, 177, 2, 69, 186, 174, 124, 153, 45},
            // 009905 这种车票和保险单捆绑出售属于重复性购买。
            {262, 44, 262, 163, 39, 41, 173, 99, 71, 42, 37, 28, 260, 84, 40, 14, 179, 152, 220, 37, 21, 39, 183, 177, 170, 179, 177, 185, 240, 39, 162, 69, 186, 260, 128, 70, 170, 154, 9},
            // 009906 戴佩妮的男友西米露接唱情歌，让她非常开心。
            {40, 10, 173, 49, 155, 72, 40, 45, 155, 15, 142, 260, 72, 154, 74, 153, 186, 179, 151, 103, 39, 22, 174, 126, 70, 41, 179, 175, 22, 182, 2, 69, 46, 39, 20, 152, 7, 260, 120},
            // 009907 观大势、谋大局、出大策始终是该院的办院方针。
            {70, 199, 40, 5, 177, 116, 154, 168, 40, 5, 151, 240, 179, 39, 183, 40, 5, 38, 44, 179, 177, 115, 262, 161, 177, 116, 70, 7, 247, 40, 45, 37, 17, 247, 69, 19, 262, 51},
            // 009908 他们骑着摩托回家，正好为农忙时的父母帮忙。
            {182, 2, 154, 55, 174, 73, 262, 45, 154, 157, 182, 230, 71, 212, 151, 77, 180, 262, 59, 71, 29, 214, 155, 162, 154, 20, 177, 114, 40, 45, 69, 186, 154, 185, 37, 19, 154, 20},
            // 009909 但是因为还没到退休年龄，只能掰着指头捱日子。
            {40, 17, 177, 116, 120, 214, 71, 8, 154, 47, 40, 30, 182, 214, 260, 140, 155, 83, 153, 126, 180, 262, 115, 155, 57, 37, 7, 262, 45, 262, 115, 182, 171, 8, 175, 116, 261, 112},
            // 009910 这几天雨水不断，人们恨不得待在家里不出门。
            {262, 44, 151, 74, 182, 82, 240, 177, 213, 37, 184, 40, 202, 180, 175, 52, 154, 55, 71, 54, 37, 186, 40, 42, 40, 7, 261, 10, 151, 77, 153, 74, 37, 186, 39, 183, 154, 52}

    };

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.btn_play:
                if (!mediaPlayer.isPlaying()) {
                    mediaPlayer.start();
                }
                break;
            case R.id.btn_pause:
                if (mediaPlayer.isPlaying()) {
                    mediaPlayer.pause();
                }
                break;
            case R.id.btn_stop:
                if (mediaPlayer.isPlaying()) {
                    mediaPlayer.reset();
                    initMediaPlayer();
                }
                break;
            default:
                break;
        }
    }

    private void initMediaPlayer() {
        try {
            File file = new File(wavFile);
            // 指定音频文件的路径
            mediaPlayer.setDataSource(file.getPath());
            // 让 MediaPlayer 进入到准备状态
            mediaPlayer.prepare();
            // 该方法使得进入应用时就播放音频
            // mediaPlayer.setOnPreparedListener(this);
            // prepare async to not block main thread
            mediaPlayer.prepareAsync();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onPrepared(MediaPlayer player) {
        player.start();
    }

    @Override
    public boolean onError(MediaPlayer mp, int what, int extra) {
        // The MediaPlayer has moved to the Error state, must be reset!
        mediaPlayer.reset();
        initMediaPlayer();
        return true;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        requestAllPermissions();
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 初始化控件
        Spinner spinner = findViewById(R.id.spinner1);
        // 建立数据源
        String[] sentences = getResources().getStringArray(R.array.text);
        // 建立 Adapter 并且绑定数据源
        ArrayAdapter<String> adapter = new ArrayAdapter<String>(this, android.R.layout.simple_spinner_dropdown_item, sentences);
        // 第一个参数表示在哪个 Activity 上显示，第二个参数是系统下拉框的样式，第三个参数是数组。
        spinner.setAdapter(adapter);//绑定Adapter到控件
        spinner.setOnItemSelectedListener(this);

        btn_play = findViewById(R.id.btn_play);
        btn_pause = findViewById(R.id.btn_pause);
        btn_stop = findViewById(R.id.btn_stop);

        btn_play.setOnClickListener(this);
        btn_pause.setOnClickListener(this);
        btn_stop.setOnClickListener(this);

        btn_play.setVisibility(View.INVISIBLE);
        btn_pause.setVisibility(View.INVISIBLE);
        btn_stop.setVisibility(View.INVISIBLE);


        // Clear all setting items to avoid app crashing due to the incorrect settings
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.clear();
        editor.commit();

        // Prepare the worker thread for mode loading and inference
        receiver = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case RESPONSE_LOAD_MODEL_SUCCESSED:
                        pbLoadModel.dismiss();
                        onLoadModelSuccessed();
                        break;
                    case RESPONSE_LOAD_MODEL_FAILED:
                        pbLoadModel.dismiss();
                        Toast.makeText(MainActivity.this, "Load model failed!", Toast.LENGTH_SHORT).show();
                        onLoadModelFailed();
                        break;
                    case RESPONSE_RUN_MODEL_SUCCESSED:
                        pbRunModel.dismiss();
                        onRunModelSuccessed();
                        break;
                    case RESPONSE_RUN_MODEL_FAILED:
                        pbRunModel.dismiss();
                        Toast.makeText(MainActivity.this, "Run model failed!", Toast.LENGTH_SHORT).show();
                        onRunModelFailed();
                        break;
                    default:
                        break;
                }
            }
        };

        worker = new HandlerThread("Predictor Worker");
        worker.start();
        sender = new Handler(worker.getLooper()) {
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case REQUEST_LOAD_MODEL:
                        // Load model and reload test image
                        if (onLoadModel()) {
                            receiver.sendEmptyMessage(RESPONSE_LOAD_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_LOAD_MODEL_FAILED);
                        }
                        break;
                    case REQUEST_RUN_MODEL:
                        // Run model if model is loaded
                        if (onRunModel()) {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_FAILED);
                        }
                        break;
                    default:
                        break;
                }
            }
        };

        // Setup the UI components
        tvInputSetting = findViewById(R.id.tv_input_setting);
        tvInferenceTime = findViewById(R.id.tv_inference_time);
        tvInputSetting.setMovementMethod(ScrollingMovementMethod.getInstance());
    }

    @Override
    protected void onResume() {
        super.onResume();
        boolean settingsChanged = false;
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        String model_path = sharedPreferences.getString(getString(R.string.MODEL_PATH_KEY),
                getString(R.string.MODEL_PATH_DEFAULT));

        settingsChanged |= !model_path.equalsIgnoreCase(modelPath);

        int cpu_thread_num = Integer.parseInt(sharedPreferences.getString(getString(R.string.CPU_THREAD_NUM_KEY),
                getString(R.string.CPU_THREAD_NUM_DEFAULT)));
        settingsChanged |= cpu_thread_num != cpuThreadNum;
        String cpu_power_mode =
                sharedPreferences.getString(getString(R.string.CPU_POWER_MODE_KEY),
                        getString(R.string.CPU_POWER_MODE_DEFAULT));
        settingsChanged |= !cpu_power_mode.equalsIgnoreCase(cpuPowerMode);

        if (settingsChanged) {
            modelPath = model_path;
            cpuThreadNum = cpu_thread_num;
            cpuPowerMode = cpu_power_mode;
            // Update UI
            tvInputSetting.setText("Model: " + modelPath.substring(modelPath.lastIndexOf("/") + 1) + "\n" + "CPU" +
                    " Thread Num: " + cpuThreadNum + "\n" + "CPU Power Mode: " + cpuPowerMode + "\n");
            tvInputSetting.scrollTo(0, 0);
            // Reload model if configure has been changed
            loadModel();
        }
    }

    public void loadModel() {
        pbLoadModel = ProgressDialog.show(this, "", "Loading model...", false, false);
        sender.sendEmptyMessage(REQUEST_LOAD_MODEL);
    }

    public void runModel() {
        pbRunModel = ProgressDialog.show(this, "", "Running model...", false, false);
        sender.sendEmptyMessage(REQUEST_RUN_MODEL);
    }

    public boolean onLoadModel() {
        return predictor.init(MainActivity.this, modelPath, AMmodelName, VOCmodelName, cpuThreadNum,
                cpuPowerMode);
    }

    public boolean onRunModel() {
        return predictor.isLoaded() && predictor.runModel(phones);
    }

    public boolean onLoadModelSuccessed() {
        // Load test image from path and run model
//        runModel();
        return true;
    }

    public void onLoadModelFailed() {
    }

    public void onRunModelSuccessed() {
        // Obtain results and update UI
        btn_play.setVisibility(View.VISIBLE);
        btn_pause.setVisibility(View.VISIBLE);
        btn_stop.setVisibility(View.VISIBLE);
        tvInferenceTime.setText("Inference done！\nInference time: " + predictor.inferenceTime() + " ms"
                + "\nRTF: " + predictor.inferenceTime() * sampleRate / (predictor.wav.length * 1000) + "\nAudio saved in " + wavFile);
        try {
            Utils.rawToWave(wavFile, predictor.wav, sampleRate);
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        } else {
            // 初始化 MediaPlayer
            initMediaPlayer();
        }
    }

    public void onRunModelFailed() {
    }


    public void onSettingsClicked() {
        startActivity(new Intent(MainActivity.this, SettingsActivity.class));
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_action_options, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                finish();
                break;
            case R.id.settings:
                onSettingsClicked();
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {

        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
        }
    }


    @Override
    protected void onDestroy() {
        if (predictor != null) {
            predictor.releaseModel();
        }
        worker.quit();
        super.onDestroy();
        if (mediaPlayer != null) {
            mediaPlayer.stop();
            mediaPlayer.release();
        }
    }

    private boolean requestAllPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED || ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    0);
            return false;
        }
        return true;
    }


    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
        if (position > 0) {
            phones = sentencesToChoose[position - 1];
            runModel();
        }

    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {

    }
}
