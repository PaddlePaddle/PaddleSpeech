<template>
    <div class="iebox">
        <h1>信息抽取体验</h1>
        <el-button :type="recoType" @click="startRecorder()"  style="margin:1vw;">{{ recoText }}</el-button>
        <h3>识别结果: {{ asrResultOffline }}</h3>
        <h4>时间：{{ time }}</h4>
        <h4>出发地：{{ outset }}</h4>
        <h4>目的地：{{ destination }}</h4>
        <h4>费用：{{ amount }}</h4>

    </div>
</template>

<script>
import Recorder from 'js-audio-recorder'

const recorder = new Recorder({
  sampleBits: 16,                 // 采样位数，支持 8 或 16，默认是16
  sampleRate: 16000,              // 采样率，支持 11025、16000、22050、24000、44100、48000，根据浏览器默认值，我的chrome是48000
  numChannels: 1,                 // 声道，支持 1 或 2， 默认是1
  compiling: true
})

    export default {
        name: "IE",
        data(){
            return {
                streamAsrResult: '',
                recoType: "primary",
                recoText: "开始录音",
                playType: "success",
                asrResultOffline: '',
                onReco: false,
                ws:'',

                time: '',
                outset: '',
                destination: '',
                amount: ''

            }
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
                
                this.time = ''
                this.outset=''
                this.destination = ''
                this.amount = ''

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
            async uploadFile(file, post_url){
                const formData = new FormData()
                formData.append('files', file)
                const result = await this.$http.post(post_url, formData);
                if (result.data.code === 0) {
                    this.asrResultOffline = result.data.result
                    this.$nextTick(()=>{})
                    this.$message.success(result.data.message);
                    this.informationExtract()
                } else {
                    this.$message.error(result.data.message);
                }
            },
            async informationExtract(){
                const postdata = {
                    chat: this.asrResultOffline
                }
                const result = await this.$http.post('/api/nlp/ie', postdata)
                console.log("ie", result)

                                if(result.data.result[0]['时间']){
                    this.time = result.data.result[0]['时间'][0]['text']
                }
                
                if(result.data.result[0]['出发地']){
                    this.outset = result.data.result[0]['出发地'][0]['text']
                }

                if(result.data.result[0]['目的地']){
                    this.destination = result.data.result[0]['目的地'][0]['text']
                }

                if(result.data.result[0]['费用']){
                    this.amount = result.data.result[0]['费用'][0]['text']
                }
            }

        },

        
    }
</script>

<style lang="less" scoped>
 .iebox {
  border: 4px solid #F00;
  top:80%;
  width: 100%;
  height: 20%;
  overflow: auto;
 }
</style>