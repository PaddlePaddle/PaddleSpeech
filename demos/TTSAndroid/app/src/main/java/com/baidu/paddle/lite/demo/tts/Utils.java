package com.baidu.paddle.lite.demo.tts;

import static java.lang.Math.abs;

import android.content.Context;
import android.os.Environment;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class Utils {
    public static void copyFileFromAssets(Context appCtx, String srcPath, String dstPath) {
        if (srcPath.isEmpty() || dstPath.isEmpty()) {
            return;
        }
        InputStream is = null;
        OutputStream os = null;
        try {
            is = new BufferedInputStream(appCtx.getAssets().open(srcPath));
            os = new BufferedOutputStream(new FileOutputStream(new File(dstPath)));
            byte[] buffer = new byte[1024];
            int length = 0;
            while ((length = is.read(buffer)) != -1) {
                os.write(buffer, 0, length);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                os.close();
                is.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static void copyDirectoryFromAssets(Context appCtx, String srcDir, String dstDir) {
        if (srcDir.isEmpty() || dstDir.isEmpty()) {
            return;
        }
        try {
            if (!new File(dstDir).exists()) {
                new File(dstDir).mkdirs();
            }
            for (String fileName : appCtx.getAssets().list(srcDir)) {
                String srcSubPath = srcDir + File.separator + fileName;
                String dstSubPath = dstDir + File.separator + fileName;
                if (new File(srcSubPath).isDirectory()) {
                    copyDirectoryFromAssets(appCtx, srcSubPath, dstSubPath);
                } else {
                    copyFileFromAssets(appCtx, srcSubPath, dstSubPath);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static String getSDCardDirectory() {
        return Environment.getExternalStorageDirectory().getAbsolutePath();
    }

    public static void rawToWave(String file, float[] data, int samplerate) throws IOException {
        // creating the empty wav file.
        File waveFile = new File(file);
        waveFile.createNewFile();
        //following block is converting raw to wav.
        DataOutputStream output = null;
        try {
            output = new DataOutputStream(new FileOutputStream(waveFile));
            // WAVE header
            // chunk id
            writeString(output, "RIFF");
            // chunk size
            writeInt(output, 36 + data.length * 2);
            // format
            writeString(output, "WAVE");
            // subchunk 1 id
            writeString(output, "fmt ");
            // subchunk 1 size
            writeInt(output, 16);
            // audio format (1 = PCM)
            writeShort(output, (short) 1);
            // number of channels
            writeShort(output, (short) 1);
            // sample rate
            writeInt(output, samplerate);
            // byte rate
            writeInt(output, samplerate * 2);
            // block align
            writeShort(output, (short) 2);
            // bits per sample
            writeShort(output, (short) 16);
            // subchunk 2 id
            writeString(output, "data");
            // subchunk 2 size
            writeInt(output, data.length * 2);
            short[] short_data = FloatArray2ShortArray(data);
            for (int i = 0; i < short_data.length; i++) {
                writeShort(output, short_data[i]);
            }
        } finally {
            if (output != null) {
                output.close();
            }
        }
    }

    private static void writeInt(final DataOutputStream output, final int value) throws IOException {
        output.write(value);
        output.write(value >> 8);
        output.write(value >> 16);
        output.write(value >> 24);
    }

    private static void writeShort(final DataOutputStream output, final short value) throws IOException {
        output.write(value);
        output.write(value >> 8);
    }

    private static void writeString(final DataOutputStream output, final String value) throws IOException {
        for (int i = 0; i < value.length(); i++) {
            output.write(value.charAt(i));
        }
    }

    public static short[] FloatArray2ShortArray(float[] values) {
        float mmax = (float) 0.01;
        short[] ret = new short[values.length];

        for (int i = 0; i < values.length; i++) {
            if (abs(values[i]) > mmax) {
                mmax = abs(values[i]);
            }
        }

        for (int i = 0; i < values.length; i++) {
            values[i] = values[i] * (32767 / mmax);
            ret[i] = (short) (values[i]);
        }
        return ret;
    }

}
