<template>
    <div class="realTime">
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
            实时得到识别结果
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
import { apiURL } from '../../../../api/API'

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
            wsUrl: "",
            ws: ""
        }
    },
    mounted () {
        this.wsUrl = apiURL.ASR_SOCKET_RECORD
        this.ws = new WebSocket(this.wsUrl)
        if(this.ws.readyState === this.ws.CONNECTING){
            this.$message.success("实时识别 Websocket 连接成功")
        }
        var _that = this
        this.ws.addEventListener('message', function (event) {
                var temp = JSON.parse(event.data);
                // console.log('ws message', event.data)
                if(temp.result && (temp.result != _that.streamAsrResult)){
                    _that.asrResult = temp.result
                    _that.$nextTick(()=>{})
                }                
        })

    },
    methods: {
        // 开始录音
        startRecorder(){
            // 检查 websocket 状态
            // debugger
            if(this.ws.readyState != this.ws.OPEN){
                this.$message.error("websocket 链接失败，请检查链接地址是否正确")
                return
            }

            this.onReco = true

            // 先跟后端说开始
            var start = JSON.stringify({name:"test.wav", "nbest":5, signal:"start"})
            this.ws.send(start)

            recorder.start().then(() => {
                setInterval(() => {
                // 持续录音
                let newData = recorder.getNextData();
                if (!newData.length) {
                    return;
                }
                // 上传到流式测试1
                this.uploadChunk(newData)
                }, 300)
            }, (error) => {
            console.log("录音出错");
            })
            // this.onReco = true
        },
        
        // 停止录音
        endRecorder(){
            // 结束录音
            recorder.stop()
            this.onReco = false
            recorder.clear()
        },

        // 流式上传
        uploadChunk(chunkDatas){
                chunkDatas.forEach((chunkData) => {
                this.ws.send(chunkData)
            })
        },
    },

}
</script>

<style lang="less" scoped>
@import "./style.less";
</style>