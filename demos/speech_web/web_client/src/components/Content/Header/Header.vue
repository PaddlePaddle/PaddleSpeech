<template>
<div className="speech_header">
      <div className="speech_header_title">
        飞桨-PaddleSpeech
      </div>
      <div className="speech_header_describe">
        PaddleSpeech 是基于飞桨 PaddlePaddle 的语音方向的开源模型库，用于语音和音频中的各种关键任务的开发，欢迎大家Star收藏鼓励
      </div>
      <div className="speech_header_link_box">
        <a href="https://github.com/PaddlePaddle/PaddleSpeech" className="speech_header_link"  target='_blank' rel='noreferrer' key={index}>
            前往Github
        </a>
      </div>
    </div>

</template>

<script>
export default {
    name:"Header"
}
</script>

<style lang="less" scoped>
@import "./style.less";
</style>