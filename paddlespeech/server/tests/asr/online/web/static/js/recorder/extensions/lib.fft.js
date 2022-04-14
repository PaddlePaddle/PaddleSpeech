/*
录音
https://github.com/xiangyuecn/Recorder
src: extensions/lib.fft.js
*/
Recorder.LibFFT=function(r){"use strict";var s,v,d,l,F,b,g,m;return function(r){var o,t,a,f;for(s=Math.round(Math.log(r)/Math.log(2)),d=((v=1<<s)<<2)*Math.sqrt(2),l=[],F=[],b=[0],g=[0],m=[],o=0;o<v;o++){for(a=o,f=t=0;t!=s;t++)f<<=1,f|=1&a,a>>>=1;m[o]=f}var n,u=2*Math.PI/v;for(o=(v>>1)-1;0<o;o--)n=o*u,g[o]=Math.cos(n),b[o]=Math.sin(n)}(r),{transform:function(r){var o,t,a,f,n,u,e,h,M=1,i=s-1;for(o=0;o!=v;o++)l[o]=r[m[o]],F[o]=0;for(o=s;0!=o;o--){for(t=0;t!=M;t++)for(n=g[t<<i],u=b[t<<i],a=t;a<v;a+=M<<1)e=n*l[f=a+M]-u*F[f],h=n*F[f]+u*l[f],l[f]=l[a]-e,F[f]=F[a]-h,l[a]+=e,F[a]+=h;M<<=1,i--}t=v>>1;var c=new Float64Array(t);for(n=-(u=d),o=t;0!=o;o--)e=l[o],h=F[o],c[o-1]=n<e&&e<u&&n<h&&h<u?0:Math.round(e*e+h*h);return c},bufferSize:v}};