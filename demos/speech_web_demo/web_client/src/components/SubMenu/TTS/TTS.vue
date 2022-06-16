<template>
    <div class="speech_recognition">
      <!-- {/* 中文文本 */} -->
      <div class="recognition_text">
        <div class="recognition_text_header">
          <div class="recognition_text_title">
            中文文本
          </div>
          <div class="recognition_text_random" @click="getRandomChineseWord()">
            <span></span><span>更换示例</span>
          </div>
        </div>

        <div class="recognition_text_field">

            <el-input
            v-model="textarea"
            :autosize="{ minRows: 13, maxRows: 13 }"
            type="textarea"
            placeholder="Please input"
            />

            
        </div>
      </div>
      <!-- {/* 指向 */} -->
      <div class="recognition_point_to"></div>
      <!-- {/* 语音合成 */} -->
      <div class="speech_recognition_new">
        <div class="speech_recognition_title">
          语音合成
        </div>
            <!-- 流式合成初始状态 -->
            <div  v-if="streamingOnInit" class="speech_recognition_streaming"
              @click="getTtsChunkWavWS()"
            >
              流式合成
            </div>
            <!-- 流式合成播放状态 -->
            <div v-else>
                <div v-if="streamingStopStatus" class="streaming_ing_box">
                <div class="streaming_ing">
                <div class="streaming_ing_img"></div>
                <!-- <Spin indicator={antIcon} /> -->
                <div class="streaming_ing_text">合成中</div>
                </div>
                <div class="streaming_time">响应时间：0ms</div>
            </div>
            <div v-else>
                <div v-if="streamingContinueStatus" class="streaming_suspended_box">
                    <div class="streaming_suspended"
                    @click="streamingStop()"
                    >
                    <div class="streaming_suspended_img"></div>
                    <div class="streaming_suspended_text">暂停播放</div>

                    </div>
                    <div class="suspended_time">
                    响应时间：{{ Number(streamingAcceptStamp) - Number(streamingSendStamp) }}ms
                    </div>
                </div>
                <div v-else class="streaming_continue"
                    @click="streamingResume()"
                >
                    <div class="streaming_continue_img"></div>
                    <div class="streaming_continue_text">继续播放</div>
                </div>
                </div>
            </div>
         
            


            <!-- //  {/* 端到端合成 */} -->
            <div v-if="endToEndOnInit" class="speech_recognition_end_to_end"
              @click="EndToEndSynthesis()"
            >
              端到端合成
            </div>
            <div v-else>
                <div  v-if="endToEndStopStatus"  class="end_to_end_ing_box">
                  <div class="end_to_end_ing">
                    <div class="end_to_end_ing_img"> </div>
                    <!-- <Spin indicator={antIcon}></Spin> -->
                    <div class="end_to_end_ing_text">合成中</div>

                  </div>
                  <div class="end_to_end_ing_time">响应时间：0s</div>
                </div>
                
                <div v-else class="end_to_end_suspended_box">
                    <div v-if="endToEndContinueStatus" class="end_to_end_suspended"
                        @onClick="EndToEndStop()"
                    >
                    <div class="end_to_end_suspended_img"></div>
                    <div class="end_to_end_suspended_text">暂停播放</div>

                    </div>
                    <div v-else class="end_to_end_continue"
                      @click="EndToEndResume()"
                    >
                      <div class="end_to_end_continue_img"></div>
                      <div class="end_to_end_continue_text">继续播放</div>
                    </div>
                    <div class="end_to_end_ing_suspended_time">响应时间：{{Number(endToEndAcceptStamp) - Number(endToEndSendStamp) }}ms</div>
                </div>
            </div>
                
      </div>
    </div>
</template>

<script>
import Recorder from 'js-audio-recorder'

// 全局承接流式 chunk 块
let chunks = []
let AudioContext = window.AudioContext || window.webkitAudioContext;
let chunk_index = 0
let palyIndex = 0
let reciveOver = false


// 定义新的流式播放服务
let _audioSrcNodes = []
const _audioCtx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
let _playStartedAt = 0
let _totalTimeScheduled = 0

function _reset(){
    _playStartedAt = 0
    _totalTimeScheduled = 0
    _audioSrcNodes = []
}



export default {
    name: "TTSTS",
    data () {
        return {
            textarea: "",
            audioCtx: '',
            source: '',
            typedArray: '',
            ttsResult: '',
            ws: '',

            // 控制播放状态
            streamingContinueStatus: true,
            endToEndContinueStatus: true,
            // 控制初始状态
            streamingOnInit: true, 
            endToEndOnInit: true, 
            // 控制是否开始
            streamingStopStatus: false,
            endToEndStopStatus: false,

            // 流式接收时间戳
            streamingAcceptStamp: '0',
            endToEndAcceptStamp: '0',
            // 流式发起时间戳
            streamingSendStamp: '0',
            endToEndSendStamp: '0'
            
        }
    },
    mounted(){
        this.getRandomChineseWord()
        
        this.ws = new WebSocket("ws://10.21.226.174:8010/ws/tts/online")
        var _that = this
        this.ws.addEventListener('message', function (event) {
            let temp = JSON.parse(event.data);
            if(chunk_index === 0){
                _that.streamingStopStatus = false
                _that.streamingAcceptStamp = Date.now()
            }

            // 接收的数据刷进播放器
            if(!temp.done){
                chunk_index += 1
                let chunk = temp.wav
                let arraybuffer = _that.base64ToUint8Array(chunk)
                let view = new DataView(arraybuffer.buffer);
                
                let length = view.buffer.byteLength / 2
                
                view = Recorder.encodeWAV(view, 24000, 24000, 1, 16, true) 
                _that._schedulePlaybackWav({
                    wavData: view.buffer,
                })
            } else {
                reciveOver = true
                // this.streamingOnInit = true
            }})
    },

    methods: {
        // 状态变量重置
        resetStatus(){
            this.streamingContinueStatus = true
            this.streamingOnInit = true
            this.streamingStopStatus = false

            this.endToEndContinueStatus = true
            this.endToEndOnInit = true
            this.endToEndStopStatus = false
        },

        // 生成随机文本
        getRandomChineseWord(){
            const resultChina = [
                "钱伟长想到上海来办学校是经过深思熟虑的。",
                "林荒大吼出声，即便十年挣扎，他也从未感到过如此无助。自己的身体一点点陷入岁月之门，却眼睁睁的看着君倾城一手持剑，雪白的身影决然凄厉。就这样孤身一人，于漫天风雪中，对阵数千武者。",
                "我们将继续成长，用行动回击那些只会说风凉话，不愿意和我们相向而行的害群之马。",
                "许多道理，人们已经证明过千遍万遍，为什么还要带着侥幸的心理再去试验一回呢？",
                "宫内整洁利索，廊柱门窗颜色鲜艳，几名电工正在维修线路。",
                "他身材矮小，颧骨突出，留着小胡子，说话一口浓重的福建口音。",
                "阿杰让阿悦看下剩下的盒饭合不合他的胃口。",
                "有网友问，能不能回忆几件刘洋在学校里的趣事或糗事。"
                ];
            let text = "";

            text = resultChina[Math.floor(Math.random() * 7)];
            this.textarea = text
        },
        // 基于WS的流式合成
        async getTtsChunkWavWS(){
            // 初始化 chunks
            chunks = []
            chunk_index = 0
            reciveOver = false
            _reset()
            
            this.streamingOnInit = false
            this.streamingStopStatus = true
            this.streamingContinueStatus = true

            this.streamingSendStamp = Date.now()
            this.ws.send(this.textarea)
        },
        // 流式播放器
        _schedulePlaybackWav({wavData}) {
            var _that = this
            _audioCtx.decodeAudioData(wavData, audioBuffer => {
            const audioSrc = _audioCtx.createBufferSource()
            audioSrc.onended = () => {
                _audioSrcNodes.shift();
                if(_audioSrcNodes.length === 0){
                    _that.resetStatus()
                }
                };
            _audioSrcNodes.push(audioSrc);
            let startDelay = 0;
            if (!_playStartedAt) {
                startDelay = 10 / 1000;
                _playStartedAt = _audioCtx.currentTime + startDelay;
                }
            audioSrc.buffer = audioBuffer;
            audioSrc.connect(_audioCtx.destination);
            
            const startAt = _playStartedAt + _totalTimeScheduled;
            audioSrc.start(startAt);

            _totalTimeScheduled+= audioBuffer.duration;

            })    
        },

        // base64转换
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
        
        // 暂停播放
        playerPaused(){
            _audioCtx.suspend()
        },

        // 恢复播放
        playerResume(){
            _audioCtx.resume()
        },

        // 流式播放暂停
        streamingStop(){
            this.playerPaused()
            // 切换为暂停状态
            this.streamingContinueStatus = false

        },
        // 流式播放恢复
        streamingResume(){
            this.playerResume()
            this.streamingContinueStatus = true
        },
        
        // 端到端合成
        async EndToEndSynthesis(){
            this.endToEndSendStamp = Date.now()
            this.endToEndOnInit = false
            this.endToEndStopStatus = true

            let ttsResult = await this.$http.post("/api/tts/offline", { text : this.textarea});
            
            if (ttsResult.status == 200) {
                this.endToEndAcceptStamp = Date.now()
                this.endToEndStopStatus = false
                this.endToEndContinueStatus = true
                // base转换二进制数
                console.log('res', ttsResult)
                let typedArray = this.base64ToUint8Array(ttsResult.data.result)
                // 播放音频
                this._schedulePlaybackWav({
                    wavData: typedArray.buffer,
                })                
            };
        },

        // 端到端播放暂停
        streamingStop(){
            this.playerPaused()
            // 切换为暂停状态
            this.endToEndContinueStatus = false

        },
        // 端到端播放恢复
        streamingResume(){
            this.playerResume()
            this.endToEndContinueStatus = true
        },




    }

}

</script>



<style lang="less" scoped>
.speech_recognition {
    width: 1200px;
    height: 410px;
    background: #FFFFFF;
    padding: 40px 0px 50px 50px;
    box-sizing: border-box;
    display: flex;
    .recognition_text {
        width: 589px; 
        height: 320px;
        // background: pink;
       .recognition_text_header {
           margin-bottom: 30px;
           display: flex;
           justify-content: space-between;
           align-items: center;
           .recognition_text_title {
            height: 26px;
            font-family: PingFangSC-Medium;
            font-size: 16px;
            color: #000000;
            letter-spacing: 0;
            line-height: 26px;
            font-weight: 500;
           };
           .recognition_text_random {
               display: flex;
               align-items: center;
               cursor: pointer;
               span {
                display: inline-block;
                    &:nth-of-type(1) {
                        width: 20px;
                        height: 20px;
                        background: url("../../../assets/image/ic_更换示例.svg") no-repeat;
                        background-position: center;
                        background-size: 20px 20px;
                        margin-right: 5px;
                       
                    };
                    &:nth-of-type(2) {
                        height: 20px;
                        font-family: PingFangSC-Regular;
                        font-size: 14px;
                        color: #2932E1;
                        letter-spacing: 0;
                        font-weight: 400;
                    };
               };
           };
       };
       .recognition_text_field {
        width: 589px;
        height: 264px;
        background: #FAFAFA;
        .textToSpeech_content_show_text{
            width: 100%;
            height: 264px;
            padding: 0px 30px 30px 0px;
            box-sizing: border-box;
            .ant-input {
                height: 208px;
                resize: none;
                // margin-bottom: 230px;
                padding: 21px 20px;
            };
        };
       };
    };
    // 指向
    .recognition_point_to {
        width: 47px;
        height: 63px;
        background: url("../../../assets/image/步骤-箭头切图@2x.png") no-repeat;
        background-position: center;
        background-size: 47px 63px;
        margin-top: 164px;
        margin-right: 101px;
        margin-left: 100px;
        margin-top: 164px;
       };
    //    语音合成
    .speech_recognition_new {
        .speech_recognition_title {
            height: 26px;
            font-family: PingFangSC-Medium;
            font-size: 16px;
            color: #000000;
            line-height: 26px;
            font-weight: 500;
            margin-left: 32px;
            margin-bottom: 96px;
        };
        // 流式合成
        .speech_recognition_streaming {
            width: 136px;
            height: 44px;
            background: #2932E1;
            border-radius: 22px;
            font-family: PingFangSC-Medium;
            font-size: 14px;
            color: #FFFFFF;
            font-weight: 500;
            text-align: center;
            line-height: 44px;
            margin-bottom: 40px;
            cursor: pointer;
            &:hover {
                opacity: .9;
            };
        };
         // 合成中
        .streaming_ing_box {
            display: flex;
            align-items: center;
            height: 44px;
            margin-bottom: 40px;
         .streaming_ing {
            width: 136px;
            height: 44px;
            background: #7278F5;
            border-radius: 22px;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
          
            .streaming_ing_img {
                width: 16px;
                height: 16px;
                // background: url("../../../assets/image/ic_小-录制语音.svg");
                // background-repeat: no-repeat;
                // background-position: center;
                // background-size: 16px 16px;
                // margin-right: 12px;
            };
            .streaming_ing_text {
                height: 20px;
                font-family: PingFangSC-Medium;
                font-size: 14px;
                color: #FFFFFF;
                font-weight: 500;
                margin-left: 12px;
            };
        };
        // 合成时间文字
        .streaming_time {
            height: 20px;
            font-family: PingFangSC-Medium;
            font-size: 14px;
            color: #000000;
            font-weight: 500;
            margin-left: 12px;
        };
        };
        

        // 暂停播放
        .streaming_suspended_box {
            display: flex;
            align-items: center;
            height: 44px;
            margin-bottom: 40px;
            .streaming_suspended {
                width: 136px;
                height: 44px;
                background: #2932E1;
                border-radius: 22px;
                display: flex;
                justify-content: center;
                align-items: center;
                cursor: pointer;
              
                .streaming_suspended_img {
                    width: 16px;
                    height: 16px;
                    background: url("../../../assets/image/ic_暂停（按钮）.svg");
                    background-repeat: no-repeat;
                    background-position: center;
                    background-size: 16px 16px;
                    margin-right: 12px;
                };
                .streaming_suspended_text {
                    height: 20px;
                    font-family: PingFangSC-Medium;
                    font-size: 14px;
                    color: #FFFFFF;
                    font-weight: 500;
                    margin-left: 12px;
                };
               
            };
            // 暂停获取时间
            .suspended_time {
                height: 20px;
                font-family: PingFangSC-Medium;
                font-size: 14px;
                color: #000000;
                font-weight: 500;
                margin-left: 12px;
            }
        };
       
        // 继续播放
        .streaming_continue {
            width: 136px;
            height: 44px;
            background: #2932E1;
            border-radius: 22px;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            margin-bottom: 40px;
            .streaming_continue_img {
                width: 16px;
                height: 16px;
                background: url("../../../assets/image/ic_播放（按钮）.svg");
                background-repeat: no-repeat;
                background-position: center;
                background-size: 16px 16px;
                margin-right: 12px;
            };
            .streaming_continue_text {
                height: 20px;
                font-family: PingFangSC-Medium;
                font-size: 14px;
                color: #FFFFFF;
                font-weight: 500;
            };
        };






        // 端到端合成
        .speech_recognition_end_to_end {
            width: 136px;
            height: 44px;
            background: #2932E1;
            border-radius: 22px;
            font-family: PingFangSC-Medium;
            font-size: 14px;
            color: #FFFFFF;
            font-weight: 500;
            text-align: center;
            line-height: 44px;
            cursor: pointer;
            &:hover {
                opacity: .9;
            };
        };
        // 合成中
        .end_to_end_ing_box {
            display: flex;
            align-items: center;
            height: 44px;
            .end_to_end_ing {
                width: 136px;
                height: 44px;
                background: #7278F5;
                border-radius: 22px;
                display: flex;
                justify-content: center;
                align-items: center;
                cursor: pointer;
                .end_to_end_ing_img {
                    width: 16px;
                    height: 16px;
                    // background: url("../../../assets/image/ic_小-录制语音.svg");
                    // background-repeat: no-repeat;
                    // background-position: center;
                    // background-size: 16px 16px;
                   
                };
                .end_to_end_ing_text {
                    height: 20px;
                    font-family: PingFangSC-Medium;
                    font-size: 14px;
                    color: #FFFFFF;
                    font-weight: 500;
                    margin-left: 12px;
                };
            };
            // 合成时间文本
            .end_to_end_ing_time {
                height: 20px;
                font-family: PingFangSC-Medium;
                font-size: 14px;
                color: #000000;
                font-weight: 500;
                margin-left: 12px;
            };
        };
       

        // 暂停播放
        .end_to_end_suspended_box {
            display: flex;
            align-items: center;
            height: 44px;
            .end_to_end_suspended {
                width: 136px;
                height: 44px;
                background: #2932E1;
                border-radius: 22px;
                display: flex;
                justify-content: center;
                align-items: center;
                cursor: pointer;
                .end_to_end_suspended_img {
                    width: 16px;
                    height: 16px;
                    background: url("../../../assets/image/ic_暂停（按钮）.svg");
                    background-repeat: no-repeat;
                    background-position: center;
                    background-size: 16px 16px;
                    margin-right: 12px;
                };
                .end_to_end_suspended_text {
                    height: 20px;
                    font-family: PingFangSC-Medium;
                    font-size: 14px;
                    color: #FFFFFF;
                    font-weight: 500;
                };
            };
            // 暂停播放时间
            .end_to_end_ing_suspended_time {
                height: 20px;
                font-family: PingFangSC-Medium;
                font-size: 14px;
                color: #000000;
                font-weight: 500;
                margin-left: 12px;
            };
        };
       
        // 继续播放
        .end_to_end_continue {
            width: 136px;
            height: 44px;
            background: #2932E1;
            border-radius: 22px;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            .end_to_end_continue_img {
                width: 16px;
                height: 16px;
                background: url("../../../assets/image/ic_播放（按钮）.svg");
                background-repeat: no-repeat;
                background-position: center;
                background-size: 16px 16px;
                margin-right: 12px;
            };
            .end_to_end_continue_text {
                height: 20px;
                font-family: PingFangSC-Medium;
                font-size: 14px;
                color: #FFFFFF;
                font-weight: 500;
            };
        };
    };
};
</style>