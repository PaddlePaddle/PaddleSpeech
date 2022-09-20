<template>
    <div class="sat">
      <el-row :gutter="20">
            <el-col :span="12"><div class="grid-content ep-bg-purple" />
                <el-row :gutter="60" class="btn_row_wav" justify="center">
                    <el-button class="ml-3" v-if="onEnrollRec === 0" @click="startRecorderEnroll()" type="primary">录制音频</el-button>
                    <el-button class="ml-3" v-else-if="onEnrollRec === 1" @click="stopRecorderEnroll()" type="danger">停止录音</el-button>
                    <el-button class="ml-3" v-else @click="uploadRecord()" type="success">上传录音</el-button>
                    <a>&#12288</a>
                    <el-upload
                        :multiple="false"
                        :accept="'.wav'"
                        :auto-upload="false"
                        :on-change="handleChange"
                        :show-file-list="false"
                    >
                        <el-button class="ml-3" type="success">上传音频文件</el-button>
                    </el-upload>
                </el-row>
                <div class="recording_table">
                <el-table :data="vcDatas" border class="recording_table_box" scrollbar-always-on max-height="250px">
                    <!-- <el-table-column prop="wavId" label="序号" width="60"/> -->
                    <el-table-column prop="wavName" label="文件名" width="150"/>
                    <el-table-column label="文本">
                      <template #default="scope">
                            <el-input 
                              v-model="scope.row.label"
                              :autosize="{ minRows: 8, maxRows: 13 }" 
                              placeholder="Please input"
                              />
                            
                        </template>
                    </el-table-column>
                    <el-table-column label="操作" width="80">
                        <template #default="scope">
                            <div class="flex justify-space-between mb-4 flex-wrap gap-4">
                                <a @click="PlayTable(scope.row.wavId)"><el-icon><VideoPlay /></el-icon></a>
                                <a>&#12288</a>
                                <a @click="delWav(scope.row.wavId)"><el-icon><DeleteFilled /></el-icon></a>
                            </div>
                        </template>
                    </el-table-column>
                    <el-table-column fixed="right" label="选择" width="70">
                        <template #default="scope">
                            <el-switch v-model="scope.row.status"  @click="choseWav(scope.row.wavId)"/>
                        </template>
                    </el-table-column>
                </el-table>
                </div>

            </el-col>
            <el-col :span="8"><div class="grid-content ep-bg-purple" />
                <el-space direction="vertical">
                    <el-card class="box-card" style="width: 250px; height:310px">
                        <template #header>
                            <div class="card-header">
                            <span>功能选择</span>
                            </div>
                        </template>  
                        <el-radio-group v-model="funcMode">
                          <el-radio label="1" size="middle" border style="margin-bottom: 10px">个性化语音合成</el-radio>
                            <el-input
                              v-if="funcMode === '1'"
                              v-model="ttsText"
                              :autosize="{ minRows: 2, maxRows: 2 }"
                              type="textarea"
                              placeholder="Please input"
                              style="margin-bottom: 10px"
                              />
                          <el-radio label="2" size="middle" border style="margin-bottom: 10px">跨语言语音合成</el-radio>
                            <el-input
                              v-if="funcMode === '2'"
                              v-model="ttsText"
                              :autosize="{ minRows: 2, maxRows: 2 }"
                              type="textarea"
                              placeholder="Please input"
                              style="margin-bottom: 10px"
                              />
                          <el-radio label="3" size="middle" border style="margin-bottom: 10px">语音编辑</el-radio>
                            <el-input
                                v-if="funcMode === '3'"
                                v-model="ttsText"
                                :autosize="{ minRows: 2, maxRows: 2 }"
                                type="textarea"
                                placeholder="Please input"
                                style="margin-bottom: 10px"
                                />
                        </el-radio-group>
                    </el-card>                    
                </el-space>
            </el-col>
            <el-col :span="4"><div class="grid-content ep-bg-purple" />
                <div class="play_board">
                    <el-space direction="vertical">
                        <el-row :gutter="20">
                            <el-button size="large" v-if="onSyn === 0" type="primary" @click="SatSyn()">开始合成</el-button>
                            <el-button size="large" v-else :loading-icon="Eleme" type="danger">合成中</el-button>
                        </el-row>
                        <el-row :gutter="20">
                            <el-button v-if='this.cloneWav' type="success" @click="PlaySyn()">播放</el-button>
                            <el-button v-else disabled type="success" @click="PlaySyn()">播放</el-button>
                            <el-button v-if='this.cloneWav' type="primary" @click="downLoadCloneWav()">下载</el-button>
                            <el-button v-else disabled type="primary" @click="downLoadCloneWav()">下载</el-button>
                        </el-row>
                    </el-space>
                </div>
            </el-col>
        </el-row>
</div>
</template>

<script>
import { vcCloneSAT, vcDownload, vcDownloadBase64, satUpload, satList, vcDel } from '../../../api/ApiVC'
import Recorder from 'js-audio-recorder'

let audioCtx = new AudioContext({
latencyHint: 'interactive',
sampleRate: 24000,
});

// 初始化录音
const recorder = new Recorder({
  sampleBits: 16,                 // 采样位数，支持 8 或 16，默认是16
  sampleRate: 16000,              // 采样率，支持 11025、16000、22050、24000、44100、48000，根据浏览器默认值，我的chrome是48000
  numChannels: 1,                 // 声道，支持 1 或 2， 默认是1
  compiling: true
})

export default {
name:"",
data(){
    return {
        uploadStatus : 0,
        recognitionStatus : 0,
        asrResult : "",
        indicator : "",
        
        filename: "",
        upfile: "",
        mode: 1,
        language: 1,
        wav_input: "卡尔普陪外孙玩滑梯",
        new_input: "卡尔普陪外孙打滑梯",
        received_file:"",

        // 分割线
        onEnrollRec: 0,
        onSyn:0,
        vcDatas: [],
        funcMode: '1',
        selected_Id: -1,
        ttsText: '',
        cloneWav: '',
        wav:''
    }
},

mounted () {
        this.GetList()
    },

methods:{
    // 获取文件列表
    async GetList(){
            this.vcDatas =[]
            const result = await satList();
            console.log("List: ", result);
            for(let i=0; i < result.data.result.length; i++){
                this.vcDatas.push({
                    wavName: result.data.result[i]['name'],
                    wavId: i,
                    wavPath: result.data.result[i]['path'],
                    status: false,
                    label: result.data.result[i]['label']
                })
            }
            console.log("vcDatas: ", this.vcDatas);
            this.$nextTick(()=>{})
    },

    // 上传文件切换
    async handleChange(file, fileList){
      for(let i=0; i<fileList.length; i++){
        this.uploadFile(fileList[i])
      }
      this.GetList()
    },

    async uploadFile(file){
      let formData = new FormData();
      formData.append('files', file.raw);
      const result = await satUpload(formData);
      if (result.data.code === 0) {
          this.$message.success("音频上传成功")
          
      } else {
          this.$message.error("音频上传失败")
      }
    },

    // 开始录音
    startRecorderEnroll(){
            this.onEnrollRec = 1
            recorder.clear()
            recorder.start()
        },
    
    // 结束录音
    stopRecorderEnroll(){
        this.onEnrollRec = 2
        recorder.stop()
        this.wav = recorder.getWAVBlob()
    },

    // 上传录音
    async uploadRecord(){
            this.onEnrollRec = 0
            if(this.wav === ""){
                this.$message.error("未检测到录音，录音失败，请重新录制")
                return
            } else {
                if(this.wav === ''){
                    this.$message.error("请先完成录音");
                    this.onEnrollRec = 0
                    return
                } else {
                    let formData = new FormData();
                    formData.append('files', this.wav);
                    const result = await satUpload(formData);
                    console.log(result)
                    this.GetList() 
                }
                this.$message.success("录音上传成功")
            }
        }, 

    // 删除音频文件
    async delWav(wavId){
            console.log('wavId', wavId)
            // 删除文件
            const result = await vcDel(
                {
                  wavName: this.vcDatas[wavId]['wavName'],
                  wavPath: this.vcDatas[wavId]['wavPath']
                }
            );
            if(!result.data.code){
                this.$message.success("删除成功")
            } else {
                this.$message.error(result.data.msg)
            }
            this.GetList()
            this.reset()
        },
    
    // 播放表格
    async PlayTable(wavId){
        this.Play(this.vcDatas[wavId])
    },

    // 播放音频
    async Play(wavBase){
        // 获取音频数据
        const result = await vcDownloadBase64(wavBase);
        // console.log('play result', result)
        if (result.data.code === 0) {
            // base转换二进制数
            let typedArray = this.base64ToUint8Array(result.data.result)
            // 添加wav文件头
            let view = new DataView(typedArray.buffer);
            view = Recorder.encodeWAV(view, 16000, 16000, 1, 16, true);
            // 播放音频
            this.playAudioData(view.buffer);
        };
        },
    // chose wav
    choseWav(wavId){
            this.cloneWav = ''
            this.nowFile = this.vcDatas[wavId].wavName
            this.nowIndex = wavId
            // only wavId is true else false
            for(let i=0; i<this.vcDatas.length; i++){
                if(i==wavId){
                    this.vcDatas[wavId].status = true
                    this.selected_Id = wavId
                    this.ttsText = this.vcDatas[wavId]['label']
                } else {
                    this.vcDatas[i].status = false
                }
            }
            this.$nextTick(()=>{})
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

    // 检查是否包含中文
    hasChinese(str) {
      return /[\u4E00-\u9FA5]+/g.test(str)
    },

    // SAT合成
    async SatSyn(){
      // 检查 select id
      if(this.selected_Id < 0){
        return this.$message.error("请先选择音频文件！")
      }

      // 检查音频对应的文本
      if(!this.vcDatas[this.selected_Id]['label']){
        return this.$message.error("音频对应文本不可以为空！")
      }

      // 检查待合成文本
      if(!this.ttsText){
        return this.$message.error("合成文本不可以为空！")
      }

      // 合成中
      this.onSyn = 1
      // 重置 clone wav
      this.cloneWav = ""
  
      const old_str = this.vcDatas[this.selected_Id]['label']
      const new_str = this.ttsText
      let language = ""
      // 包含中文
      if(this.hasChinese(old_str)){
        language = "zh"
      } else{
        language = "en"
      }
      // 功能选择
      let func = ""
      if(this.funcMode === '1') {
        func = "synthesize"
      } else if(this.funcMode === '2'){
        func = "crossclone"
      } else {
        func = "edit"
      }
      
      let wav_path = this.vcDatas[this.selected_Id]['wavPath']
      let filename = this.vcDatas[this.selected_Id]['wavName']

      const data = {
        old_str: old_str,
        new_str: new_str,
        language: language,
        function: func,
        wav: wav_path,
        filename: filename

      }

      console.log("sat data: ", data)
      
      // sat 接口
      const result = await vcCloneSAT(data)
      // 合成完成
      this.onSyn = 0
      console.log(result);
      // debugger
      if (result.data.code === 0) {

        this.$message.success(result.data.message)
        // 获取识别文本
        this.cloneWav = result.data.result
        console.log("cloneWave", this.cloneWav);

      } else {
        this.$message.error(result.data.message)
      };
    },
    // 播放合成的音频
    // 播放音频
    async PlaySyn(){
        // 获取音频数据
        const data = {
          wavName: "sat_"+this.filename,
          wavPath: this.cloneWav
        }
        const result = await vcDownloadBase64(data);
        // console.log('play result', result)
        if (result.data.code === 0) {
            // base转换二进制数
            let typedArray = this.base64ToUint8Array(result.data.result)
            // 添加wav文件头
            let view = new DataView(typedArray.buffer);
            view = Recorder.encodeWAV(view, 16000, 16000, 1, 16, true);
            // 播放音频
            this.playAudioData(view.buffer);
        };
        },


    // 下载合成文件
    async downLoadCloneWav(){
    if(this.cloneWav  === ""){
        this.$message.error("音频合成完毕后再下载！")
    } else {
        // const result = await vcDownload(this.cloneWav);
        // 获取音频数据
        const data = {
          wavName: "sat_"+this.filename,
          wavPath: this.cloneWav
        }
        const result = await vcDownloadBase64(data);
        let view;
        // console.log('play result', result)
        if (result.data.code === 0) {
            // base转换二进制数
            let typedArray = this.base64ToUint8Array(result.data.result)
            // 添加wav文件头
            view = new DataView(typedArray.buffer);
            view = Recorder.encodeWAV(view, 16000, 16000, 1, 16, true);
            // 播放音频
            // this.playAudioData(view.buffer);
        }
        console.log(view.buffer)
        // debugger
        const blob = new Blob([view.buffer], { type: 'audio/wav' });
        const fileName = new Date().getTime() + '.wav';
        const down = document.createElement('a');
        down.download = fileName;
        down.style.display = 'none';//隐藏,没必要展示出来
        down.href = URL.createObjectURL(blob);
        document.body.appendChild(down);
        down.click();
        URL.revokeObjectURL(down.href); // 释放URL 对象
        document.body.removeChild(down);//下载完成移除
      }
    },

}
}   

</script>

<style lang="less" scoped>
// @import "./style.less";
.sat {
    width: 1200px;
    height: 410px;
    background: #FFFFFF;
    padding: 5px 80px 56px 80px;
    box-sizing: border-box;
}

.el-row {
  margin-bottom: 20px;
}
.grid-content {
  border-radius: 4px;
  min-height: 36px;
}
.play_board{
    height: 100%;
    display: flex;
    align-items: center;
}

</style>