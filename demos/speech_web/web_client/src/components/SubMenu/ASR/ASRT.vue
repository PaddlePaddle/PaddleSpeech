<script setup>
import AudioFileIdentification from "./AudioFile/AudioFileIdentification.vue"
import RealTime from "./RealTime/RealTime.vue"
import EndToEndIdentification from "./EndToEnd/EndToEndIdentification.vue";


</script>

<template>
    <div class="speech_recognition">
      <div class="speech_recognition_tabs">
        <div class="frame"></div>
        <el-tabs class="speech_recognition_mytabs" type="border-card">
          <el-tab-pane label="实时语音识别" key="1">
            <RealTime />
          </el-tab-pane>
          <el-tab-pane label="端到端识别" key="2">
            <EndToEndIdentification />
          </el-tab-pane>
          <el-tab-pane label="音频文件识别" key="3">
            <AudioFileIdentification />
          </el-tab-pane>
        </el-tabs>
      </div>
    </div>
</template>

<script>

export default {
    
}
</script>

<style lang="less" scoped>
@import "./style.less";

</style>