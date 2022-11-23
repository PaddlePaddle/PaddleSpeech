export const apiURL =   {
    ASR_OFFLINE : '/api/asr/offline',           // 获取离线语音识别结果
    ASR_COLLECT_ENV : '/api/asr/collectEnv',    // 采集环境噪音
    ASR_STOP_RECORD : '/api/asr/stopRecord',    // 后端暂停录音
    ASR_RESUME_RECORD : '/api/asr/resumeRecord',// 后端恢复录音

    NLP_CHAT : '/api/nlp/chat',                 // NLP闲聊接口
    NLP_IE : '/api/nlp/ie',                     // 信息抽取接口

    TTS_OFFLINE : '/api/tts/offline',           // 获取TTS音频

    VPR_RECOG : '/api/vpr/recog',               // 声纹识别接口，返回声纹对比相似度
    VPR_ENROLL : '/api/vpr/enroll',             // 声纹识别注册接口
    VPR_LIST : '/api/vpr/list',                 // 获取声纹注册的数据列表
    VPR_DEL : '/api/vpr/del',                   // 删除用户声纹
    VPR_DATA : '/api/vpr/database64?vprId=',            // 获取声纹注册数据 bs64格式

    // websocket
    CHAT_SOCKET_RECORD: 'ws://localhost:8010/ws/asr/offlineStream', // ChatBot websocket 接口
    ASR_SOCKET_RECORD: 'ws://localhost:8010/ws/asr/onlineStream',  // Stream ASR 接口
    TTS_SOCKET_RECORD: 'ws://localhost:8010/ws/tts/online', // Stream TTS 接口

    // voice clone
    // Voice Clone
    VC_List: '/api/vc/list',
    SAT_List: '/api/sat/list',
    FineTune_List: '/api/finetune/list',

    VC_Upload: '/api/vc/upload',
    SAT_Upload: '/api/sat/upload',
    FineTune_Upload: '/api/finetune/upload',
    FineTune_NewDir: '/api/finetune/newdir',

    VC_Download: '/api/vc/download',
    VC_Download_Base64: '/api/vc/download_base64',
    VC_Del: '/api/vc/del',
    
    VC_CloneG2p: '/api/vc/clone_g2p',
    VC_CloneSAT: '/api/vc/clone_sat',
    VC_CloneFineTune: '/api/finetune/clone_finetune',
    VC_CloneFineTuneSyn: '/api/finetune/clone_finetune_syn',
}







