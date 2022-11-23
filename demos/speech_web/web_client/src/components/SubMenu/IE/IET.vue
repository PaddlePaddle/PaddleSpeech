<template>
    <div class="voice_commands">
      <div class="voice_commands_traffic">
        <div class="voice_commands_traffic_title">交通费报销</div>
        <div class="voice_commands_traffic_wrapper">
          <div class="voice_commands_traffic_wrapper_move">
            <div class="traffic_btn_img_btn">
                <!-- 结束录音 -->
                <div v-if="onReco"
                @click="endRecorder()"
                class="end_recorder_img"
                ></div>
                <!-- 开始录音 -->
                <div v-else
                @click= "startRecorder()"
                class="start_recorder_img"
                ></div>
            </div>
            <div class="traffic_btn_prompt">
                <div v-if="onReco">
                    结束识别
                </div>
                <div v-else>
                    开始识别
                </div>
            </div>
            <div class="traffic_btn_list">试试说“早上八点，我从广州到北京花了四百二十六元”</div>
          </div>
        </div>
      </div>

      <div class="voice_point_to"></div>

      <!-- 识别结果 -->
      <div class="voice_commands_IdentifyTheResults">
        <div class="voice_commands_IdentifyTheResults_title">
          识别结果
        </div>
 
        <div v-if="postStatus" class="voice_commands_IdentifyTheResults_show">
            <div class="voice_commands_IdentifyTheResults_show_title">
              {{ asrResult }}
            </div>
            <div class="oice_commands_IdentifyTheResults_show_time">
              时间：{{voiceCommandsData.time}}
            </div>
            <div class="oice_commands_IdentifyTheResults_show_money">
              费用：{{voiceCommandsData.amount}}
            </div>
            <div class="oice_commands_IdentifyTheResults_show_origin">
              出发地：{{voiceCommandsData.outset}}
            </div>
            <div class="oice_commands_IdentifyTheResults_show_destination">
              目的地：{{voiceCommandsData.destination}}
            </div>
            </div>
        <div v-else class="voice_commands_IdentifyTheResults_show_loading">
                <a-spin />
        </div>
        
      </div>
    </div >

</template>

<script>
import Recorder from 'js-audio-recorder'
import { asrOffline } from '../../../api/ApiASR'
import { nlpIE } from '../../../api/ApiNLP'

const recorder = new Recorder({
  sampleBits: 16,                 // 采样位数，支持 8 或 16，默认是16
  sampleRate: 16000,              // 采样率，支持 11025、16000、22050、24000、44100、48000，根据浏览器默认值，我的chrome是48000
  numChannels: 1,                 // 声道，支持 1 或 2， 默认是1
  compiling: true
})


export default {
  
    data () {
        return {
            voiceCommandsData:{
                time:"",
                amount:"",
                outset:"",
                destination:""
            },
            asrDeafult : "语音识别结果",
            asrResult: "",
            postStatus:true,
            onReco:false
        }
    },
    mounted () {
      this.asrResult = this.asrDeafult
    },
    methods: {
      // reset
      reset(){
          this.asrResult = this.asrDeafult
          this.voiceCommandsData = {
                  time:"",
                  amount:"",
                  outset:"",
                  destination:""
              }
         },

      // 开始录音
        startRecorder(){
          this.reset()
          this.onReco = true
          recorder.clear()
          recorder.start()
        },
      // 停止录音
        endRecorder(){
            recorder.stop()
            this.onReco = false
            // this.$nextTick(()=>{})
            this.postStatus = false
            const wavs = recorder.getWAVBlob()
            this.uploadFile(wavs)
        },
      // 上传识别结果
        async uploadFile(file){
          const formData = new FormData();
          formData.append('files', file)
          const result = await asrOffline(formData)
            if (result.data.code === 0) {
                this.asrResult = result.data.result
                this.$message.success(result.data.message);
                this.informationExtract()
            } else {
                this.$message.error(result.data.message);
            }
        },
        // 信息抽取
        async informationExtract(){
                const result = await nlpIE(this.asrResult)

                if(result.data.result[0]['时间']){
                    this.voiceCommandsData.time = result.data.result[0]['时间'][0]['text']
                }
                
                if(result.data.result[0]['出发地']){
                    this.voiceCommandsData.outset = result.data.result[0]['出发地'][0]['text']
                }

                if(result.data.result[0]['目的地']){
                    this.voiceCommandsData.destination = result.data.result[0]['目的地'][0]['text']
                }

                if(result.data.result[0]['费用']){
                    this.voiceCommandsData.amount = result.data.result[0]['费用'][0]['text']
                }
                this.postStatus = true
            }
    }
}
</script>

<style lang="less" scoped>
@import "./style.less";
</style>