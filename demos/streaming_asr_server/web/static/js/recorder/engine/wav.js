/*
录音
https://github.com/xiangyuecn/Recorder
src: engine/wav.js
*/
!function(){"use strict";Recorder.prototype.enc_wav={stable:!0,testmsg:"支持位数8位、16位（填在比特率里面），采样率取值无限制"},Recorder.prototype.wav=function(t,e,n){var r=this.set,a=t.length,o=r.sampleRate,f=8==r.bitRate?8:16,i=a*(f/8),s=new ArrayBuffer(44+i),c=new DataView(s),u=0,v=function(t){for(var e=0;e<t.length;e++,u++)c.setUint8(u,t.charCodeAt(e))},w=function(t){c.setUint16(u,t,!0),u+=2},l=function(t){c.setUint32(u,t,!0),u+=4};if(v("RIFF"),l(36+i),v("WAVE"),v("fmt "),l(16),w(1),w(1),l(o),l(o*(f/8)),w(f/8),w(f),v("data"),l(i),8==f)for(var p=0;p<a;p++,u++){var d=128+(t[p]>>8);c.setInt8(u,d,!0)}else for(p=0;p<a;p++,u+=2)c.setInt16(u,t[p],!0);e(new Blob([c.buffer],{type:"audio/wav"}))}}();