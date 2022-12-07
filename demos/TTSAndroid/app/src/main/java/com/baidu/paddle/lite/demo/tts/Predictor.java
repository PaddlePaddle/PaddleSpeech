package com.baidu.paddle.lite.demo.tts;

import android.content.Context;
import android.util.Log;

import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.PaddlePredictor;
import com.baidu.paddle.lite.PowerMode;
import com.baidu.paddle.lite.Tensor;

import java.io.File;
import java.util.Date;


public class Predictor {
    private static final String TAG = Predictor.class.getSimpleName();
    public boolean isLoaded = false;
    public int cpuThreadNum = 1;
    public String cpuPowerMode = "LITE_POWER_HIGH";
    public String modelPath = "";
    protected PaddlePredictor AMPredictor = null;
    protected PaddlePredictor VOCPredictor = null;
    protected float inferenceTime = 0;
    protected float[] wav;

    public boolean init(Context appCtx, String modelPath, String AMmodelName, String VOCmodelName, int cpuThreadNum, String cpuPowerMode) {
        // Release model if exists
        releaseModel();

        AMPredictor = loadModel(appCtx, modelPath, AMmodelName, cpuThreadNum, cpuPowerMode);
        if (AMPredictor == null) {
            return false;
        }
        VOCPredictor = loadModel(appCtx, modelPath, VOCmodelName, cpuThreadNum, cpuPowerMode);
        if (VOCPredictor == null) {
            return false;
        }
        isLoaded = true;
        return true;
    }

    protected PaddlePredictor loadModel(Context appCtx, String modelPath, String modelName, int cpuThreadNum, String cpuPowerMode) {
        // Load model
        if (modelPath.isEmpty()) {
            return null;
        }
        String realPath = modelPath;
        if (modelPath.charAt(0) != '/') {
            // Read model files from custom path if the first character of mode path is '/'
            // otherwise copy model to cache from assets
            realPath = appCtx.getCacheDir() + "/" + modelPath;
            // push model to mobile
            Utils.copyDirectoryFromAssets(appCtx, modelPath, realPath);
        }
        if (realPath.isEmpty()) {
            return null;
        }
        MobileConfig config = new MobileConfig();
        config.setModelFromFile(realPath + File.separator + modelName);
        Log.e(TAG, "File:" + realPath + File.separator + modelName);
        config.setThreads(cpuThreadNum);
        if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_HIGH")) {
            config.setPowerMode(PowerMode.LITE_POWER_HIGH);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_LOW")) {
            config.setPowerMode(PowerMode.LITE_POWER_LOW);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_FULL")) {
            config.setPowerMode(PowerMode.LITE_POWER_FULL);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_NO_BIND")) {
            config.setPowerMode(PowerMode.LITE_POWER_NO_BIND);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_RAND_HIGH")) {
            config.setPowerMode(PowerMode.LITE_POWER_RAND_HIGH);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_RAND_LOW")) {
            config.setPowerMode(PowerMode.LITE_POWER_RAND_LOW);
        } else {
            Log.e(TAG, "Unknown cpu power mode!");
            return null;
        }
        return PaddlePredictor.createPaddlePredictor(config);
    }

    public void releaseModel() {
        AMPredictor = null;
        VOCPredictor = null;
        isLoaded = false;
        cpuThreadNum = 1;
        cpuPowerMode = "LITE_POWER_HIGH";
        modelPath = "";
    }

    public boolean runModel(float[] phones) {
        if (!isLoaded()) {
            return false;
        }
        Date start = new Date();
        Tensor am_output_handle = getAMOutput(phones, AMPredictor);
        wav = getVOCOutput(am_output_handle, VOCPredictor);
        Date end = new Date();
        inferenceTime = (end.getTime() - start.getTime());
        return true;
    }

    public Tensor getAMOutput(float[] phones, PaddlePredictor am_predictor) {
        Tensor phones_handle = am_predictor.getInput(0);
        long[] dims = {phones.length};
        phones_handle.resize(dims);
        phones_handle.setData(phones);
        am_predictor.run();
        Tensor am_output_handle = am_predictor.getOutput(0);
        // [?, 80]
        // long outputShape[] = am_output_handle.shape();
        float[] am_output_data = am_output_handle.getFloatData();
        // [? x 80]
        // long[] am_output_data_shape = {am_output_data.length};
        // Log.e(TAG, Arrays.toString(am_output_data));
        // 打印 mel 数组
        // for (int i=0;i<outputShape[0];i++) {
        //      Log.e(TAG, Arrays.toString(Arrays.copyOfRange(am_output_data,i*80,(i+1)*80)));
        // }
        // voc_predictor 需要知道输入的 shape，所以不能输出转成 float 之后的一维数组
        return am_output_handle;
    }

    public float[] getVOCOutput(Tensor input, PaddlePredictor voc_predictor) {
        Tensor mel_handle = voc_predictor.getInput(0);
        // [?, 80]
        long[] dims = input.shape();
        mel_handle.resize(dims);
        float[] am_output_data = input.getFloatData();
        mel_handle.setData(am_output_data);
        voc_predictor.run();
        Tensor voc_output_handle = voc_predictor.getOutput(0);
        // [? x 300, 1]
        // long[] outputShape = voc_output_handle.shape();
        float[] voc_output_data = voc_output_handle.getFloatData();
        // long[] voc_output_data_shape = {voc_output_data.length};
        return voc_output_data;
    }


    public boolean isLoaded() {
        return AMPredictor != null && VOCPredictor != null && isLoaded;
    }


    public float inferenceTime() {
        return inferenceTime;
    }

}
