<template>
        <div class="audioFileIdentification">

      
        <div v-if="uploadStatus === 0" class="public_recognition_speech">
            <!-- 上传前 -->
            <el-upload
                :multiple="false"
                :accept="'.wav'"
                :limit="1"
                :auto-upload="false"
                :on-change="handleChange"
                :show-file-list="false"
            >
                <div class="upload_img">
                <div class="upload_img_back"></div>
                </div>
            </el-upload>
            <div class="speech_text">
                上传文件
            </div>
            <div class="speech_text_prompt">
                支持50秒内的.wav文件
            </div>
        </div>
        <!-- 上传中 -->
        <div v-else-if="uploadStatus === 1" class="on_the_cross_speech">
            <div class="on_the_upload_img">
                <div class="on_the_upload_img_back"></div>
            </div>
            <div class="on_the_speech_text">
                <span class="on_the_speech_loading"> <Spin indicator={antIcon} /></span> 上传中
            </div>
        </div>
        <div v-else>

            <!-- // {/* //开始识别 */} -->
                <div v-if="recognitionStatus === 0" class="public_recognition_speech_start">
                <div class="public_recognition_speech_content">
                  <div
                    class="public_recognition_speech_title"
                  >
                   {{ filename }}
                  </div>
                  <div
                    class="public_recognition_speech_again"
                    @click="uploadAgain()"
                  >重新上传</div>
                  <div
                    class="public_recognition_speech_play"
                    @click="paly()"
                  >播放</div>
                </div>
                <div class="speech_promp"
                    @click="beginToIdentify()">
                  开始识别
                </div>
                </div>
                <!-- //  {/* 识别中 */} -->
                <div v-else-if="recognitionStatus === 1" class="public_recognition_speech_identify">
                <div class="public_recognition_speech_identify_box">
                <div
                    class="public_recognition_speech_identify_back_img"
                  > 
                    <a-spin  />
                  </div> 

                  <div
                    class="public_recognition__identify_the_promp"
                  >识别中</div>
        
                </div>
                </div>
              
                <!-- //  {/* // 重新识别 */} -->
              <div v-else class="public_recognition_speech_identify_ahain">
              <div class="public_recognition_speech_identify_box_btn">
              
                <div
                  class="public_recognition__identify_the_btn"
                  @click="toIdentifyThe()"
                >重新识别</div>
      
              </div>
                </div>
            
        </div>

      <!-- {/* 指向 */} -->
      <div class="public_recognition_point_to">

      </div>
      <!-- {/* 识别结果 */} -->
      <div class="public_recognition_result">
        <div>识别结果</div>
        <div>{{ asrResult }}</div>
      </div>
    </div>
</template>

<script>
import { asrOffline } from '../../../../api/ApiASR'

let audioCtx = new AudioContext({
  latencyHint: 'interactive',
  sampleRate: 24000,
});

export default {
    name:"",
    data(){
        return {
            uploadStatus : 0,
            recognitionStatus : 0,
            asrResult : "",
            indicator : "",
            
            filename: "",
            upfile: ""

        }
    },

    methods:{
        // 上传文件切换
        handleChange(file, fileList){
            this.uploadStatus = 2
            this.filename = file.name
            this.upfile = file
            console.log(file)
            
            // debugger
            // var result = Buffer.from(file);

            
        },
        readFile(file) {
            return new Promise((resolve, reject) => {
                const fileReader = new FileReader();
                fileReader.onload = function () {
                    resolve(fileReader);
                };
                fileReader.onerror = function (err) {
                    reject(err);
                };
                fileReader.readAsDataURL(file);
                });
            },
        // 重新上传
        uploadAgain(){
            this.uploadStatus = 0
            this.upfile = ""
            this.filename = ""
            this.asrResult = ""
        },

        // 播放音频
        playAudioData(wav_buffer){
            audioCtx.decodeAudioData(wav_buffer, buffer => {
                let source = audioCtx.createBufferSource();
                source.buffer = buffer
            
                source.connect(audioCtx.destination);
                source.start();
            }, function (e) {
            });
        },

        // 播放本地音频
        async paly(){
            if(this.upfile){
                let fileRes = ""
                let fileString = ""
                fileRes = await this.readFile(this.upfile.raw);
                fileString = fileRes.result;
                const audioBase64type = (fileString.match(/data:[^;]*;base64,/))?.[0] ?? '';
                const isBase64 = !!fileString.match(/data:[^;]*;base64,/);
                const uploadBase64 = fileString.substr(audioBase64type.length);
                // isBase64 ? uploadBase64 : undefined
                // base转换二进制数
                let typedArray = this.base64ToUint8Array(isBase64 ? uploadBase64 : undefined)
                this.playAudioData(typedArray.buffer)
            }
        },
        base64ToUint8Array(base64String){
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

        // 开始识别
        async beginToIdentify(){
          // 识别中
          this.recognitionStatus = 1
          
          const formData = new FormData();
          formData.append('files', this.upfile.raw);
          
          const result = await asrOffline(formData)
          // 重新识别
          this.recognitionStatus = 2
          console.log(result);
          // debugger
          if (result.data.code === 0) {

            this.$message.success("识别成功")
            // 获取识别文本
            this.asrResult = result.data.result

          } else {
            this.$message.success("识别失败")
          };
        },

        // 重新识别
        toIdentifyThe(){
          // this.uploadAgain()
          this.uploadStatus = 0
          this.recognitionStatus = 0
          this.asrResult = ""
        }

    }
}   

</script>

<style lang="less" scoped>
@import "./style.less";


</style>