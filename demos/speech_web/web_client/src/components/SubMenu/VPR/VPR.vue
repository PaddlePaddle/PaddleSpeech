<template>
<div class="vprbox">
        <div>
      <h1>声纹识别展示</h1>
    <el-input
      v-model="spk_id"
      class="w-50 m-2"
      size="large"
      placeholder="spk_id"
    />
    <el-button :type="recoType" @click="startRecorder()"  style="margin:1vw;">{{ recoText }}</el-button>
    <el-button type="primary" @click="Enroll(spk_id)"  style="margin:1vw;"> 注册 </el-button>
    <el-button type="primary" @click="Recog()"  style="margin:1vw;"> 识别 </el-button>
    </div>
    <div>
        <h2>声纹得分结果</h2>
        <el-table :data="score_result" style="width: 40%">
            <el-table-column prop="spkId" label="spk_id" />
            <el-table-column prop="score" label="score" />
        </el-table>
    </div>
    <div>
        <h2>声纹数据列表</h2>
        <el-table :data="vpr_datas" style="width: 40%">
            <el-table-column prop="spkId" label="spk_id" />
            <el-table-column label="wav">
                <template #default="scope2">
                    <audio :src="'/VPR/vpr/data/?vprId='+scope2.row.vprId" controls>
                    
                    </audio>
                </template>
            </el-table-column>
            <el-table-column fixed="right" label="Operations">
                <template #default="scope">
                    <el-button @click="Del(scope.row.spkId)" type="text" size="small">Delete</el-button>
                </template>
            </el-table-column>
        </el-table>

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
        name: "VPR",
        data () {
            return {
                url_enroll: '/VPR/vpr/enroll', //注册
                url_recog: '/VPR/vpr/recog',  //识别
                url_del: '/VPR/vpr/del',    // 删除
                url_list: '/VPR/vpr/list',   // 获取列表
                url_data: '/VPR/vpr/data',   // 获取音频

                spk_id: 'sss',
                onRecord: false,
                recoType: "primary",
                recoText: "开始录音",
                wav: '',

                score_result: [],
                vpr_datas: []
            }
        },
        mounted () {
            this.GetList()
        },
        methods: {
            startRecorder () {
                this.score_result = []
                if(!this.onReco){
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
                    this.wav = recorder.getWAVBlob()
                }
            },
            async Enroll(spk_id){
                if(this.wav === ''){
                    this.$message.error("请先完成录音");
                    return
                }
                let formData = new FormData()
                formData.append('spk_id', this.spk_id)
                formData.append('audio', this.wav)

                console.log("formData", formData)
                console.log("spk_id", this.spk_id)
                const result = await this.$http.post(this.url_enroll, formData);
                if(result.data.status){
                    this.$message.success("声纹注册成功")
                } else {
                    this.$message.error(result.data.msg)
                }
                console.log(result)
                this.GetList()
            },
            async Recog(){
                this.score_result = []
                if(this.wav === ''){
                    this.$message.error("请先完成录音");
                    return
                }
                let formData = new FormData()
                formData.append('audio', this.wav)
                const result = await this.$http.post(this.url_recog, formData);
                console.log(result)
                result.data.forEach(dat => {
                    this.score_result.push({
                        spkId: dat[0],
                        score: dat[1][1]
                    })
                });
            },
            async Del(spkId){
                console.log('spkId', spkId)
                // 删除用户
                const result = await this.$http.post(this.url_del, {spk_id: spkId});
                if(result.data.status){
                    this.$message.success("删除成功")
                } else {
                    this.$message.error(result.data.msg)
                }
                this.GetList()
            },
            async GetList(){
                this.vpr_datas =[]
                const result = await this.$http.get(this.url_list);
                console.log("list", result)
                for(let i=0; i<result.data[0].length; i++){
                    this.vpr_datas.push({
                        spkId: result.data[0][i],
                        vprId: result.data[1][i]
                    })
                }
                this.$nextTick(()=>{})
            },
            GetData(){},
        },

    }
</script>

<style lang='less' scoped>
.vprbox {
  border: 4px solid #F00;
//   position: fixed;
  top:60%;
  width: 100%;
  height: 20%;
  overflow: auto;
 }
</style>