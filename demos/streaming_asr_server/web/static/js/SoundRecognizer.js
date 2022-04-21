SoundRecognizer = {
    rec: null,
    wave: null,
    SampleRate: 16000,
    testBitRate: 16,
    isCloseRecorder: false,
    SendInterval: 300,
    realTimeSendTryType: 'pcm',
    realTimeSendTryEncBusy: 0,
    realTimeSendTryTime: 0,
    realTimeSendTryNumber: 0,
    transferUploadNumberMax: 0,
    realTimeSendTryChunk: null,
    soundType: "pcm",
    init: function (config) {
        this.soundType = config.soundType || 'pcm';
        this.SampleRate = config.sampleRate || 16000;
        this.recwaveElm = config.recwaveElm || '';
        this.TransferUpload = config.translerCallBack || this.TransferProcess;
        this.initRecorder();
    },
    RealTimeSendTryReset: function (type) {
        this.realTimeSendTryType = type;
        this.realTimeSendTryTime = 0;
    },
    RealTimeSendTry: function (rec, isClose) {
        var that = this;
        var t1 = Date.now(), endT = 0, recImpl = Recorder.prototype;
        if (this.realTimeSendTryTime == 0) {
            this.realTimeSendTryTime = t1;
            this.realTimeSendTryEncBusy = 0;
            this.realTimeSendTryNumber = 0;
            this.transferUploadNumberMax = 0;
            this.realTimeSendTryChunk = null;
        }
        if (!isClose && t1 - this.realTimeSendTryTime < this.SendInterval) {
            return;//控制缓冲达到指定间隔才进行传输
        }
        this.realTimeSendTryTime = t1;
        var number = ++this.realTimeSendTryNumber;

        //借用SampleData函数进行数据的连续处理，采样率转换是顺带的
        var chunk = Recorder.SampleData(rec.buffers, rec.srcSampleRate, this.SampleRate, this.realTimeSendTryChunk, { frameType: isClose ? "" : this.realTimeSendTryType });

        //清理已处理完的缓冲数据，释放内存以支持长时间录音，最后完成录音时不能调用stop，因为数据已经被清掉了
        for (var i = this.realTimeSendTryChunk ? this.realTimeSendTryChunk.index : 0; i < chunk.index; i++) {
            rec.buffers[i] = null;
        }
        this.realTimeSendTryChunk = chunk;

        //没有新数据，或结束时的数据量太小，不能进行mock转码
        if (chunk.data.length == 0 || isClose && chunk.data.length < 2000) {
            this.TransferUpload(number, null, 0, null, isClose);
            return;
        }
        //实时编码队列阻塞处理
        if (!isClose) {
            if (this.realTimeSendTryEncBusy >= 2) {
                console.log("编码队列阻塞，已丢弃一帧", 1);
                return;
            }
        }
        this.realTimeSendTryEncBusy++;

        //通过mock方法实时转码成mp3、wav
        var encStartTime = Date.now();
        var recMock = Recorder({
            type: this.realTimeSendTryType
            , sampleRate: this.SampleRate //采样率
            , bitRate: this.testBitRate //比特率
        });
        recMock.mock(chunk.data, chunk.sampleRate);
        recMock.stop(function (blob, duration) {
            that.realTimeSendTryEncBusy && (that.realTimeSendTryEncBusy--);
            blob.encTime = Date.now() - encStartTime;

            //转码好就推入传输
            that.TransferUpload(number, blob, duration, recMock, isClose);
        }, function (msg) {
            that.realTimeSendTryEncBusy && (that.realTimeSendTryEncBusy--);
            //转码错误？没想到什么时候会产生错误！
            console.log("不应该出现的错误:" + msg, 1);
        });
    },
    recordClose: function () {
        try {
            this.rec.close(function () {
                this.isCloseRecorder = true;
            });
            this.RealTimeSendTry(this.rec, true);//最后一次发送
        } catch (ex) {
            // recordClose();
        }
    },
    recordEnd: function () {
        try {
            this.rec.stop(function (blob, time) {
                this.recordClose();
            }, function (s) {
                this.recordClose();
            });
        } catch (ex) {
        }
    },
    initRecorder: function () {
        var that = this;
        var rec = Recorder({
            type: that.soundType
            , bitRate: that.testBitRate
            , sampleRate: that.SampleRate
            , onProcess: function (buffers, level, time, sampleRate) {
                that.wave.input(buffers[buffers.length - 1], level, sampleRate);
                that.RealTimeSendTry(rec, false);//推入实时处理，因为是unknown格式，这里简化函数调用，没有用到buffers和bufferSampleRate，因为这些数据和rec.buffers是完全相同的。
            }
        });

        rec.open(function () {
            that.wave = Recorder.FrequencyHistogramView({
                elem: that.recwaveElm, lineCount: 90
                , position: 0
                , minHeight: 1
                , stripeEnable: false
            });
            rec.start();
            that.isCloseRecorder = false;
            that.RealTimeSendTryReset(that.soundType);//重置
        });
        this.rec = rec;
    },
    TransferProcess: function (number, blobOrNull, duration, blobRec, isClose) {

    }
}