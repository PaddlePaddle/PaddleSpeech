<template>
    <div class="endToEndIdentification">
      <div  class="public_recognition_speech">
      
      <div v-if="onReco"> 
        <!-- 结束录音 -->
        <div @click="endRecorder()" class="endToEndIdentification_end_recorder_img">
              <div class='endToEndIdentification_end_recorder_img_back'></div>
        </div>
      </div>
      <div v-else>
        <div @click="startRecorder()" class="endToEndIdentification_start_recorder_img"></div>
      </div>
      
        <div class="endToEndIdentification_prompt" >
            <div v-if="onReco">
                结束识别
            </div>
            <div v-else>
                开始识别
            </div>
        </div>
        <div class="speech_text_prompt">
            停止录音后得到识别结果
        </div>
      </div>
      <div class="public_recognition_point_to"></div>
      <div class="public_recognition_result">
        <div>识别结果</div>
        <div> {{asrResult}} </div>
      </div>
    </div>
</template>

<script>
import Recorder from 'js-audio-recorder'
import { asrOffline } from '../../../../api/ApiASR'

const recorder = new Recorder({
  sampleBits: 16,                 // 采样位数，支持 8 或 16，默认是16
  sampleRate: 16000,              // 采样率，支持 11025、16000、22050、24000、44100、48000，根据浏览器默认值，我的chrome是48000
  numChannels: 1,                 // 声道，支持 1 或 2， 默认是1
  compiling: true
})

export default {
    data () {
        return {
            onReco: false,
            asrResult: "",
        }
    },
    methods: {
        // 开始录音
        startRecorder(){
            this.onReco = true
            recorder.clear()
            recorder.start()
        },

        // 停止录音
        endRecorder(){
            recorder.stop()
            this.onReco = false
            // this.$nextTick(()=>{})
            // 音频导出成wav,然后上传到服务器
            const wavs = recorder.getWAVBlob()
            this.uploadFile(wavs)
        },
        
        // 上传文件
         async uploadFile(file){
            const formData = new FormData()
            formData.append('files', file)
            const result = await asrOffline(formData)
            if (result.data.code === 0) {
                this.asrResult = result.data.result
                // this.$nextTick(()=>{})
                this.$message.success(result.data.message);
            } else {
                this.$message.error(result.data.message);
            }
        },

    }
    
}
</script>

<style lang="less" scoped>
@import "./style.less";
</style>