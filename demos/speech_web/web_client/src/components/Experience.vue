<script setup>

import ChatT from './SubMenu/ChatBot/ChatT.vue'
import ASRT from './SubMenu/ASR/ASRT.vue'
import TTST from './SubMenu/TTS/TTST.vue'
import VPRT from './SubMenu/VPR/VPRT.vue'
import IET from './SubMenu/IE/IET.vue'

</script>

<template>
    <div className="experience">
      <div className="experience_wrapper">
        <div className="experience_title">
          功能体验
        </div>
        <div className="experience_describe">
          体验前，请允许浏览器获取麦克风权限
        </div>
        <div className="experience_content" >
          <el-tabs
            className="experience_tabs"
            type="border-card"
          >
            <el-tab-pane label="语音聊天" key="1">
              <ChatT></ChatT>
            </el-tab-pane>
            <el-tab-pane label="声纹识别" key="2">
             <VPRT></VPRT>
            </el-tab-pane>
            <el-tab-pane label="语音识别" key="3">
            <ASRT></ASRT>
            </el-tab-pane>
            <el-tab-pane label="语音合成" key="4">
            <TTST></TTST>
            </el-tab-pane>
            <el-tab-pane label="语音指令" key="5">
            <IET></IET>
            </el-tab-pane>
          </el-tabs>
        </div>
      </div>
    </div>
</template>


<style lang="less">
@import "./style.less";

</style>