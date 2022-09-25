<template>
    <div class="finetune">
      <el-row :gutter="20"> 
        <el-col :span="12"><div class="grid-content ep-bg-purple" />
          <el-row :gutter="60" class="btn_row_wav" justify="center">
              <el-button class="ml-3" @click="clearAll()" type="primary">一键重置</el-button>
              <el-button class="ml-3" @click="resetDefault()" type="primary">默认示例</el-button>
              <el-button v-if='onFinetune === 0' class="ml-3" @click="fineTuneModel()" type="primary">一键微调</el-button>
              <el-button v-else-if='onFinetune === 1' class="ml-3" @click="fineTuneModel()" type="danger">微调中</el-button>
              <el-button v-else-if='onFinetune === 2' class="ml-3" @click="resetFinetuneBtn()" type="success">微调成功</el-button>
              <el-button v-else class="ml-3" @click="resetFinetuneBtn()" type="success">微调失败</el-button>
              <!-- <el-button class="ml-3" @click="chooseHistory()" type="warning">历史数据选择</el-button> -->
        </el-row>

        <div class="recording_table">
            <el-table :data="vcDatas" border class="recording_table_box" scrollbar-always-on max-height="250px">
                <el-table-column prop="wavId" label="序号" width="60"/>
                <el-table-column prop="text" label="文本" />
                <el-table-column label="音频" width="80">
                    <template #default="scope">
                        <a v-if="scope.row.wavPath != ''">{{ scope.row.wavName }}</a>
                        <a v-else>
                            
                            <el-button class="ml-3" v-if="onEnrollRec === 0" @click="startRecorderEnroll()" type="primary" circle>
                                <el-icon><Microphone /></el-icon>
                            </el-button>
                            <el-button class="ml-3" v-else-if="onEnrollRec === 1" @click="stopRecorderEnroll()" type="danger" circle>
                                <el-icon><Microphone /></el-icon>
                            </el-button>
                            <el-button class="ml-3" v-else @click="uploadRecord(scope.row.wavId)" type="success" circle>
                                <el-icon><Upload /></el-icon>
                            </el-button>
                        </a>
                    </template>
                </el-table-column>
                <el-table-column label="操作" width="80" fixed="right">
                    <template #default="scope">
                        <div class="flex justify-space-between mb-4 flex-wrap gap-4">
                            <a @click="PlayTable(scope.row.wavId)"><el-icon><VideoPlay /></el-icon></a>
                            <a>&#12288</a>
                            <a @click="delWav(scope.row.wavId)"><el-icon><DeleteFilled /></el-icon></a>
                        </div>
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
                                <span>试验路径</span>
                                <el-input
                                    v-model="expPath"
                                    :autosize="{ minRows: 2, maxRows: 3 }"
                                    type="textarea"
                                    placeholder="一键微调自动生成，可使用历史试验路径"
                                    />
                            </div>
                        </template>
                        <span>请输入中文文本</span>
                        <el-input
                            v-model="ttsText"
                            :autosize="{ minRows: 5, maxRows: 6 }"
                            type="textarea"
                            placeholder="请输入待合成文本"
                            />
                    </el-card>                    
                </el-space>
            </el-col>
            <el-col :span="4"><div class="grid-content ep-bg-purple" />
                <div class="play_board">
                    <el-space direction="vertical">
                        <el-row :gutter="20">
                            <el-button size="large" v-if="onSyn === 0" type="primary" @click="fineTuneSyn()">开始合成</el-button>
                            <el-button size="large" v-else :loading-icon="Eleme" type="danger">合成中</el-button>
                        </el-row>

                        <el-row :gutter="20">
                            <el-button v-if='this.cloneWav' type="success" @click="PlaySyn()">播放</el-button>
                            <el-button v-else disabled type="primary" @click="PlaySyn()">播放</el-button>
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
    import Recorder from 'js-audio-recorder'
    import { vcDownload, vcDownloadBase64, vcCloneFineTune, vcCloneFineTuneSyn, fineTuneList, vcDel, fineTuneUpload, fineTuneNewDir } from '../../../api/ApiVC';
    
    // 初始化录音
    const recorder = new Recorder({
      sampleBits: 16,                 // 采样位数，支持 8 或 16，默认是16
      sampleRate: 16000,              // 采样率，支持 11025、16000、22050、24000、44100、48000，根据浏览器默认值，我的chrome是48000
      numChannels: 1,                 // 声道，支持 1 或 2， 默认是1
      compiling: true
    })
    
    // 初始化播放器
    const audioCtx = new AudioContext({
        latencyHint: 'interactive',
        sampleRate: 16000,
    });

    function blobToDataURL(blob, callback) {
        let a = new FileReader();
        a.onload = function (e) { callback(e.target.result); }
        a.readAsDataURL(blob);
    }

    
    export default {
        data(){
            return {
              vcDatas:[],
              defaultDataPath: 'default',
              nowDataPath: '',
              expPath: '',
              wav: '',
              wav_base64: '',
              ttsText: '',
              cloneWav: '',
              
              onEnrollRec: 0,  // 录音状态
              onFinetune: 0,  // 微调状态
              onSyn: 0, // 合成状态
            }
        },
        mounted () {
            this.nowDataPath = this.defaultDataPath
            this.GetList()
            
        },
        methods: {
            // 重置 btn 
            resetFinetuneBtn(){
                this.onFinetune = 0
            },
        
        // 一键重置
        async clearAll(){
            this.vcDatas = []
            const result = await fineTuneNewDir()
            console.log("clearALL: ", result.data.result);
            this.nowDataPath = result.data.result
            this.expPath = ''
            this.onFinetune = 0
            await this.GetList()
        },
        // 显示默认
        async resetDefault(){
            this.nowDataPath = this.defaultDataPath
            await this.GetList()
            this.expPath = ''
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
        async uploadRecord(wavId){
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
                    let fileRes = ""
                    let fileString = ""
                    fileRes = await this.readFile(this.wav);
                    fileString = fileRes.result;
                    const audioBase64type = (fileString.match(/data:[^;]*;base64,/))?.[0] ?? '';
                    const isBase64 = !!fileString.match(/data:[^;]*;base64,/);
                    const uploadBase64 = fileString.substr(audioBase64type.length);
                    
                    // 上传时指定文件路径
                    const data = {
                        'wav': uploadBase64,
                        'filename': this.vcDatas[wavId]['wavName'],
                        'wav_path': this.nowDataPath
                    }

                    const result = await fineTuneUpload(data);
                    console.log(result)
                    this.GetList() 
                }
                this.$message.success("录音上传成功")
            }
        }, 
        // 读取文件和Blob
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

            // 获取文件列表
          async GetList(){
            this.vcDatas = []
            const result = await fineTuneList({
              dataPath: this.nowDataPath
            });
            console.log(result, result.data.result);
            for(let i=0; i<result.data.result.length; i++){
                this.vcDatas.push({
                  wavId: i,
                  text: result.data.result[i]['text'],
                  wavName: result.data.result[i]['name'],
                  wavPath: result.data.result[i]['path'],
                })
            }
            this.$nextTick(()=>{})
          },
                  // 播放音频
    playAudioData( wav_buffer ) {
        audioCtx.decodeAudioData(wav_buffer, buffer => {
            var source = audioCtx.createBufferSource();
            source.buffer = buffer;
            source.connect(audioCtx.destination);
            source.start();
        }, function(e) {
            Recorder.throwError(e);
            })
    },
        // base64解码
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
            // 播放表格
        async PlayTable(wavId){
            this.Play(this.vcDatas[wavId])
        },
        // 播放合成后的音频
        async PlaySyn(){
           
            if(this.cloneWav  === ""){
                this.$message.error("请合成音频后再播放！！")
                return
            } else {
                this.Play(this.cloneWav)
            }
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
                } else {
                    this.$message.error("获取音频文件失败")
                }
        },
                // 下载合成文件
        async downLoadCloneWav(){
            if(this.cloneWav  === ""){
                this.$message.error("音频合成完毕后再下载！")
            } else {
                // const result = await vcDownload(this.cloneWav);
                // 获取音频数据
                const result = await vcDownloadBase64(this.cloneWav);
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
        // 删除音频文件
        async delWav(wavId){
            if(this.nowDataPath === this.defaultDataPath){
                this.$message.error("默认音频不允许删除，可以一键重置，重新录音")
                return 
            }

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
                this.GetList()
            } else {
                this.$message.error("文件删除失败")
            }
        }, 
        // 微调模型
        async fineTuneModel(){
            // 先检查是否都有录音
            for(let i=0; i < this.vcDatas.length; i++){
                if(this.vcDatas['wavPath'] === ''){
                    return this.$message.error("还有录音未完成，请先完成录音！")
                }
            }
            this.onFinetune = 1
            const result = await vcCloneFineTune(
                {
                    wav_path: this.nowDataPath,
                }
            );
            if(!result.data.code){
                this.onFinetune = 2
                this.expPath = result.data.result
                console.log("this.expPath: ", this.expPath)
                this.$message.success("小数据微调成功")
            } else {
                this.onFinetune = 3
                this.$message.error(result.data.msg)
            }
        },
        // 合成音频
        async fineTuneSyn(){
            if(!this.expPath){
                return this.$message.error("请先微调生成模型后再生成！")
            }
            // 合成
            this.onSyn = 1
            const result = await vcCloneFineTuneSyn(
                {
                    exp_path: this.expPath,
                    text: this.ttsText
                }
            );
            this.onSyn = 0
            if(!result.data.code){
                this.cloneWav = result.data.result
                console.log("clone wav: ", this.cloneWav)
                this.$message.success("音色克隆成功")
            } else {
                this.$message.error(result.data.msg)
            }
            this.$nextTick(()=>{})
        }
},
};
</script>
    
<style lang="less" scoped>
// @import "./style.less";
.finetune {
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