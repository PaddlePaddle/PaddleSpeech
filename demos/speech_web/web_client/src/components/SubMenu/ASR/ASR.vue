<template>
    <div class="asrbox">
        <h5> ASR 体验</h5>
        <div class="home" style="margin:1vw;">
            <el-button :type="recoType" @click="startRecorderChunk()"  style="margin:1vw;">{{ recoText }} (流式)</el-button>
            <el-button :type="recoType" @click="startRecorder()"  style="margin:1vw;">{{ recoText }} (端到端)</el-button>
      </div>
      <a> asr_stream: {{ streamAsrResult }}</a>
      <br>
      <a> asr_offline: {{ asrResultOffline }} </a>

    </div>
</template>

<script>
import Recorder from 'js-audio-recorder'

const recorder_chunk = new Recorder({
  sampleBits: 16,                 // 采样位数，支持 8 或 16，默认是16
  sampleRate: 16000,              // 采样率，支持 11025、16000、22050、24000、44100、48000，根据浏览器默认值，我的chrome是48000
  numChannels: 1,                 // 声道，支持 1 或 2， 默认是1
  compiling: true
})

const recorder = new Recorder({
  sampleBits: 16,                 // 采样位数，支持 8 或 16，默认是16
  sampleRate: 16000,              // 采样率，支持 11025、16000、22050、24000、44100、48000，根据浏览器默认值，我的chrome是48000
  numChannels: 1,                 // 声道，支持 1 或 2， 默认是1
  compiling: true
})

    export default {
        name: "ASR",
        data(){
            return {
                streamAsrResult: '',
                recoType: "primary",
                recoText: "开始录音",
                playType: "success",
                asrResultOffline: '',
                onReco: false,
                ws:'',
            }
        },
        mounted (){
            // 初始化ws
            this.ws = new WebSocket("ws://localhost:8010/ws/asr/onlineStream")
            // 定义消息处理逻辑
            var _that = this
            this.ws.addEventListener('message', function (event) {
                var temp = JSON.parse(event.data);
                // console.log('ws message', event.data)
                if(temp.result && (temp.result != _that.streamAsrResult)){
                    _that.streamAsrResult = temp.result
                    _that.$nextTick(()=>{})
                    console.log('更新了')
                }                
            })
        },

        methods: {
            startRecorder () {
                if(!this.onReco){
                    recorder.clear()
                    recorder.start().then(() => {
                    }, (error) => {
                    console.log("录音出错");
                })
                this.onReco = true
                this.recoType = "danger"
                this.recoText = "结束录音"
                this.$nextTick(()=>{
                })
                } else {
                // 结束录音
                    recorder.stop()
                    this.onReco = false
                    this.recoType = "primary"
                    this.recoText = "开始录音"
                    this.$nextTick(()=>{})
                    // 音频导出成wav,然后上传到服务器
                    const wavs = recorder.getWAVBlob()
                    this.uploadFile(wavs, "/api/asr/offline")
                }
            },


            startRecorderChunk() {
                if(!this.onReco){
                    // 跟后端说：开始流式传输
                    var start = JSON.stringify({name:"test.wav", "nbest":5, signal:"start"})
                    this.ws.send(start)
                    recorder_chunk.start().then(() => {
                        setInterval(() => {
                        // 持续录音
                        let newData = recorder_chunk.getNextData();
                        if (!newData.length) {
                            return;
                        }
                        // 上传到流式测试1
                        this.uploadChunk(newData)
                        }, 500)
                    }, (error) => {
                    console.log("录音出错");
                    })
                    this.onReco = true
                    this.recoType = "danger"
                    this.recoText = "结束录音"
                    this.$nextTick(()=>{
                    })
                } else {
                    // 结束录音
                    recorder_chunk.stop()
                    // 跟后端说不录了
                    // var end = JSON.stringify({name:"test.wav", "nbest":5, signal:"end"})
                    // this.ws.send(end)
                    this.onReco = false
                    this.recoType = "primary"
                    this.recoText = "开始录音"
                    this.$nextTick(()=>{})
                    recorder_chunk.clear()
                }
            },
            uploadChunk(chunkDatas){
                chunkDatas.forEach((chunkData) => {
                this.ws.send(chunkData)
              })
            },
            async uploadFile(file, post_url){
                const formData = new FormData()
                formData.append('files', file)
                const result = await this.$http.post(post_url, formData);
                if (result.data.code === 0) {
                    this.asrResultOffline = result.data.result
                    this.$nextTick(()=>{})
                    this.$message.success(result.data.message);
                } else {
                    this.$message.error(result.data.message);
                }
            },
        },
    }
</script>

<style lang='less' scoped>
 .asrbox {
  border: 4px solid #F00;
//   position: fixed;
  top:40%;
  width: 100%;
  height: 20%;
  overflow: auto;
 }
</style>