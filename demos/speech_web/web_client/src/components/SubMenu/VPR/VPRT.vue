<template>
<div class="voiceprint">
        <div class="voiceprint_recording">
            <div class="recording_title">
                <div>1</div>
                <div>
                    录制声纹
                </div>
            </div>
            <div>
                试试对我说：欢迎使用飞桨声纹识别系统
            </div>
            <!-- 开始录音 -->
                <div v-if="onEnrollRec === 0 " class="recording_btn"
                    @click="startRecorderEnroll()"
                >
                    <div class="recording_img"></div>
                        <div class="recording_prompt">
                            录制声音
                        </div>
                </div>
                <!-- 结束录音 -->
                <div v-else-if="onEnrollRec === 1 " class="recording_btn_the_recording"
                    @click="stopRecorderEnroll(0)"
                >
                    <a-spin />
                    <div class="recording_prompt">
                            停止录音
                    </div>
                </div>

                <!-- :
                //  {/* 完成录音 */} -->
                <div v-else class="complete_the_recording_btn"
                    @click="enrollVoicePrint()"
                >
                    <div class="complete_the_recording_img"></div>
                    <div class="complete_the_recording_prompt">
                        注册声纹
                    </div>
                </div>
            
            <!-- 用户名输入框 -->
            <div class="recording_input">
                <el-input v-model="enrollSpkId" class="w-50 m-2" autosize placeholder="请输入注册用户名" />
            </div>
            

            <!-- {/* table */} -->
            <div class="recording_table">

                <el-table :data="vpr_datas" border class="recording_table_box">
                    <el-table-column prop="spkId" label="用户" />
                    <el-table-column fixed="right" label="操作">
                        <template #default="scope">
                            <el-button @click="Play(scope.row.vprId)" type="text" size="small">播放</el-button>
                            <el-button @click="Del(scope.row.spkId)" type="text" size="small">删除</el-button>
                        </template>
                    </el-table-column>
                </el-table>

            </div>
        </div>

        <!-- {/* 指向 */} -->
        <div class="recording_point_to"></div>

        <!-- {/* 识别声纹 */} -->
        <div class="voiceprint_identify">
            <div class="identify_title">
                <div>2</div>
                <div>
                    识别声纹
                </div>
            </div>
            <div>
                试试对我说：请识别一下我的声音
            </div>
                    <div v-if="onRegRec === 0" class="identify_btn"
                        @click="startRecorderRecog()"
                    >
                        <div class="identify_img"></div>
                        <div class="identify_prompt">
                            录制声音
                        </div>
                    </div>

                    <div v-else-if="onRegRec === 1" class="identify_btn_the_recording"
                    @click="stopRecorderRecog()">
                        <a-spin />
                        <div class="recording_prompt">
                                停止录音
                        </div>
                           
                    </div>
                    
                    <div v-else class="identify_complete_the_recording_btn"
                        @click="Recog()">
                        <div class="identify_complete_the_recording_img"></div>
                        <div class="identify_complete_the_recording_prompt">
                            开始识别
                        </div>
                    </div>

            <div class="identify_result">
                <div class="identify_result_content">
                    <div>识别结果</div>
                    <div>{{scoreResult}}</div>
                </div>
            </div>
        </div>
</div>
</template>

<script>
import Recorder from 'js-audio-recorder'
import { vprData, vprList, vprEnroll, vprRecog, vprDel } from '../../../api/ApiVPR';

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

export default {
    data(){
        return {
            onEnrollRec: 0,     // 注册录音状态
            onRegRec:0,         // 识别录音状态

            scoreResult: "",   // 得分结果
            enrollSpkId: "",    // SpkId注册

            wav: '',            // 录音结果

            scoreResults: [],   // 得分结果
            vpr_datas: []       // 数据列表


        }
    },
    mounted () {
        this.GetList()
        this.randomSpkId()
    },
    methods: {
        // 重置
        reset(){
            this.wav = ''
            this.scoreResults = []
            this.scoreResult = ""
        },
        // random SpkName
        randomSpkId(){
            var e = 3;
            var t = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄",
            a = t.length,
            n = "";
            for (var i = 0; i < e; i++) n += t.charAt(Math.floor(Math.random() * a));
            this.enrollSpkId = n
            console.log("n", n)
        },
        // 注册声纹开始录音
        startRecorderEnroll(){
            this.onEnrollRec = 1
            recorder.clear()
            recorder.start()
        },
        // 注册声纹结束录音
        stopRecorderEnroll(){
            this,this.onEnrollRec = 2
            recorder.stop()
            this.wav = recorder.getWAVBlob()
        },

        // 识别声纹开始录音
        startRecorderRecog(){
            // this.wav = ''
            this.onRegRec = 1
            this.reset()
            recorder.clear()
            recorder.start()
        },

        // 注册声纹结束录音
        stopRecorderRecog(){
            this,this.onRegRec = 2
            recorder.stop()
            this.wav = recorder.getWAVBlob()
        },

        // 注册声纹
        async enrollVoicePrint(){
                if(this.wav === ''){
                    this.$message.error("请先完成录音");
                    this.onEnrollRec = 0
                    return
                }
                if(this.enrollSpkId === ""){
                    this.$message.error("请输入声纹用户名")
                    this.onEnrollRec = 2
                    return
                }
                this.onEnrollRec = 0

                let formData = new FormData()
                formData.append('spk_id', this.enrollSpkId)
                formData.append('audio', this.wav)
                
                const result = await vprEnroll(formData)
                if (!result){
                    this.$message.error("请检查后端服务是否正确开启")
                    return 
                }
                if(result.data.status){
                    this.$message.success("声纹注册成功")
                } else {
                    this.$message.error(result.data.msg)
                }
                this.GetList()
                this.wav = ''
                this.randomSpkId()
            },

        // 识别声纹
        async Recog(){
            this.scoreResults = []
            this.onRegRec = 0
            if(this.wav === ''){
                this.$message.error("请先完成录音");
                return
            }
            if(this.vpr_datas.length == 0){
                this.$message.error("未查询到声纹数据，请先注册");
                return
            }
            let formData = new FormData()
            formData.append('audio', this.wav)
            const result = await vprRecog(formData);
            console.log(result)
            result.data.forEach(dat => {
                this.scoreResults.push({
                    spkId: dat[0],
                    score: dat[1][1]
                })
            });
            if(this.scoreResults.length > 0){
                this.scoreResult = this.scoreResults[0]['spkId']
            }
        },

        // 删除声纹
        async Del(spkId){
                console.log('spkId', spkId)
                // 删除用户
                const result = await vprDel({spk_id: spkId});
                if(result.data.status){
                    this.$message.success("删除成功")
                } else {
                    this.$message.error(result.data.msg)
                }
                this.GetList()
            },
        
        // 获取声纹列表
        async GetList(){
            this.vpr_datas =[]
            const result = await vprList();
            console.log("list", result)
            for(let i=0; i<result.data[0].length; i++){
                this.vpr_datas.push({
                    spkId: result.data[0][i],
                    vprId: result.data[1][i]
                })
            }
            this.$nextTick(()=>{})
        },

        // 播放音频
        async Play(vprId){
                console.log('vprId', vprId)
                // 获取音频数据
                const result = await vprData(vprId);
                console.log('play result', result)
                if (result.data.code == 0) {
                    // base转换二进制数
                    let typedArray = this.base64ToUint8Array(result.data.result)

                    // 添加wav文件头
                    let view = new DataView(typedArray.buffer);
                    view = Recorder.encodeWAV(view, 16000, 16000, 1, 16, true);

                    // 播放音频
                    this.playAudioData(view.buffer);
                };
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
        }
    }
};
</script>

<style lang="less" scoped>
@import "./style.less";
</style>