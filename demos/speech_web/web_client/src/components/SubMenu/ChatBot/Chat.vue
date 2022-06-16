<template>
  <div class="chatbox">
      <h3>语音聊天</h3>
      <div class="home" style="margin:1vw;">
      <el-button :type="recoType" @click="startRecorder()"  style="margin:1vw;">{{ recoText }}</el-button>
      <!-- <el-button :type="playType" @click="playRecorder()" style="margin:1vw;"> {{ playText }}</el-button> -->
      <el-button :type="envType" @click="envRecorder()" style="margin:1vw;"> {{ envText }}</el-button>
      <!-- <el-button :type="envType" @click="getTts(ttsd)" style="margin:1vw;"> TTS </el-button> -->
      <el-button type="warning" @click="clearChat()" style="margin:1vw;"> 清空聊天</el-button>

      </div>

      <div v-for="Result in allResultList">
      <h3>{{Result}}</h3>
      </div>
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
    name: 'home',
    data () {
      return {
        recoType: "primary",
        recoText: "开始录音",
        playType: "success",
        playText: "播放录音",
        envType: "success",
        envText: "环境采样",

        asrResultList: [],
        nlpResultList: [],
        ttsResultList: [],
        allResultList: [],
        webSocketRes: "websocket",
        drawRecordId: null,

        onReco: false,
        onPlay: false,
        onRecoPause: false,
        ws: '',

        ttsd: "你的名字叫什么,你的名字叫什么,你的名字叫什么你的名字叫什么",
        audioCtx: '',
        source: '',

        typedArray: '',
        ttsResult: '',
       
      }
    },
    mounted () {
        // 播放器
        var AudioContext = window.AudioContext || window.webkitAudioContext;
        this.audioCtx = new AudioContext({
            latencyHint: 'interactive',
            sampleRate: 24000,
          });
        // 定义 play
        recorder.onplayend = () => {
        this.onPlay = false
        this.playText = "播放录音"
        this.playType = "success"
        this.$nextTick(()=>{})
      }
      // 初始化ws
      this.ws = new WebSocket("ws://localhost:8010/ws/asr/offlineStream");

      // 定义消息处理逻辑
      var _that = this
      this.ws.addEventListener('message', function (event) {
          _that.allResultList.push("asr:" + event.data)
          _that.$nextTick(()=>{})
          _that.getNlp(event.data)
      })
    },

    methods: {
      // 清空录音
      clearChat(){
        this.allResultList = []
      },
      // 开始录音
      startRecorder () {
        if(!this.onReco){
          this.resumeRecordOnline()
          recorder.start().then(() => {
            setInterval(() => {
              // 持续录音
              let newData = recorder.getNextData();
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
          recorder.stop()
          this.onReco = false
          this.recoType = "primary"
          this.recoText = "开始录音"
          this.$nextTick(()=>{})
          recorder.clear()
          // 音频导出成wav,然后上传到服务器
          // const wavs = recorder.getWAVBlob()
          // this.uploadFile(wavs, "/api/asr/offline")
          // console.log(wavs)
          // 给服务器发送停止指令, 清空缓存数据
          this.stopRecordOnline()
        }
      },

      // 开始录音
      envRecorder () {
        if(!this.onReco){
          recorder.start().then(() => {
        }, (error) => {
          console.log("录音出错");
        })
        this.onReco = true
        this.envType = "danger"
        this.envText = "结束采样"
        this.$nextTick(()=>{
          })
        } else {
          // 结束录音
          recorder.stop()
          this.onReco = false
          this.envType = "success"
          this.envText = "环境采样"
          this.$nextTick(()=>{})
          const wavs = recorder.getWAVBlob()
          this.uploadFile(wavs, "/api/asr/collectEnv")
        }
      },


      // 录音播放
      playRecorder () {
        if(!this.onPlay){
          // 播放音频
          recorder.play()
          this.onPlay = true
          this.playText = "结束播放"
          this.playType = "warning"
          this.$nextTick(()=>{})
        
        } else {
          recorder.stopPlay()
          this.onPlay = false
          this.playText = "播放录音"
          this.playType = "success"
          this.$nextTick(()=>{})
        }
      },

      // 上传录音文件
      async uploadFile(file, post_url){
        const formData = new FormData()
        formData.append('files', file)
        const result = await this.$http.post(post_url, formData);
        if (result.data.code === 0) {
              this.asrResultList.push(result.data.result)
              // this.$message.success(result.data.message);
          } else {
              this.$message.error(result.data.message);
          }
      },
      // 上传chunk语音包
      async uploadChunk(chunkDatas) {
        chunkDatas.forEach((chunkData) => {
                this.ws.send(chunkData)
              })

      },

      // 停止录音,输出成pcm
      async stopRecordOnline(){
        const result = await this.$http.get("/api/asr/stopRecord");
        if (result.data.code === 0) {
            console.log("Online 录音停止成功")
          } else {
            // console.log("chunk 发送失败")
          }
      },
      // 恢复录音，中间抛出的语音，一律不接受
      async resumeRecordOnline(){
        const result = await this.$http.get("/api/asr/resumeRecord");
        if (result.data.code === 0) {
            console.log("chunk 发送成功")
          } else {
            // console.log("chunk 发送失败")
          }
      },

      // 请求 NLP 对话结果
      async getNlp(asrText){
        
        // 录音暂停
        this.onRecoPause = true
        recorder.pause()
        this.stopRecordOnline()
        console.log('录音暂停')

        const result = await this.$http.post("/api/nlp/chat", { chat: asrText});
        if (result.data.code === 0) {
              this.allResultList.push("nlp:" + result.data.result)
              this.getTts(result.data.result)
              // this.$message.success(result.data.message);
          } else {
              this.$message.error(result.data.message);
          }
        // console.log("录音恢复")
      },

    base64ToUint8Array(base64String) {
      const padding = '='.repeat((4 - base64String.length % 4) % 4);
       const base64 = (base64String + padding)
                    .replace(/-/g, '+')
                    .replace(/_/g, '/');

       const rawData = window.atob(base64);
       const outputArray = new Uint8Array(rawData.length);

       for (let i = 0; i < rawData.length; ++i) {
            outputArray[i] = rawData.charCodeAt(i);
       }
       return outputArray;
      },

      // 合成TTS音频
      async getTts(nlpText){
        // base64
        this.ttsResult = await this.$http.post("/api/tts/offline", { text : nlpText});
        this.typedArray = this.base64ToUint8Array(this.ttsResult.data.result)
        // console.log("chat", this.typedArray.buffer)
        this.playAudioData( this.typedArray.buffer )

      },

      // play
      playAudioData( wav_buffer ) {
        this.audioCtx.decodeAudioData(wav_buffer, buffer => {
            this.source = this.audioCtx.createBufferSource();
            this.source.onended = () => {
              // 如果被暂停
              if(this.onRecoPause){
                console.log("恢复录音")
                this.onRecoPause = false
                // 客户端录音恢复
                recorder.resume()
                // 服务器录音恢复
                this.resumeRecordOnline()
              }
              
            }
            this.source.buffer = buffer;
            this.source.connect(this.audioCtx.destination);
            this.source.start();
        }, function(e) {
            Recorder.throwError(e);
        });
    }
    },
 
  }
</script>
 
<style lang='less' scoped>
 .chatbox {
  border: 4px solid #F00;
  // position: fixed;
  width: 100%;
  height: 20%;
  overflow: auto;
 }
</style>