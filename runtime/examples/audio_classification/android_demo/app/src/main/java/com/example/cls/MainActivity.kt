package com.baidu.paddlespeech.cls

import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView

class MainActivity : AppCompatActivity() {
    private lateinit var tvContent: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        tvContent = findViewById(R.id.tv_content)

        nClsCreateInstance()
        tvContent.text = nClsFeedForward()
        nClsReset()
        nClsDestroyInstance()
    }

    external fun nClsCreateInstance(): Boolean
    external fun nClsDestroyInstance(): Boolean
    external fun nClsFeedForward(): String
    external fun nClsReset(): Boolean

    companion object{
        init {
            System.loadLibrary("native-lib")
        }
    }
}