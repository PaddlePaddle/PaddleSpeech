// Copyright (c) 2014 baidu-rpc authors.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A server to receive EchoRequest and send back EchoResponse.

#include <iostream>
#ifndef _WIN32
#include <sys/time.h>
#else
#include <win_func.h>
#endif


#include <gflags/gflags.h>
#include <baas-lib-c/baas.h>
#include <baas-lib-c/giano_mock_helper.h>
#include <base/logging.h>
#include "base/base64.h"
#include <baidu/rpc/server.h>
#include <baidu/rpc/policy/giano_authenticator.h>
#include "echo.pb.h"
#include <typeinfo>
#include <stdexcept>
#include <fstream>
#include <string>
#include <com_log.h>
#include <vector>
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "recognizer.h"
#include <AudioFile.h>
#include <Audio.h>
#include <cstdlib>

using namespace std;

DEFINE_int32(port, 8857, "TCP Port of this server");
DEFINE_string(modelpath, "/home/bae/sourcecode/paddlespeech_cli", "the path of model");
DEFINE_int32(max_concurrency, 2, "Limit of request processing in parallel");


namespace example {
struct subRes{
    int s; // start time
    int e; // end time
    string t; // text
};
class AudioServiceImpl : public EchoService {
public:
    AudioServiceImpl() {};
    virtual ~AudioServiceImpl() {};
    string to_string(rapidjson::Document& out_doc) {
        rapidjson::StringBuffer out_buffer;
        rapidjson::Writer<rapidjson::StringBuffer> out_writer(out_buffer);
        out_doc.Accept(out_writer);
        std::string out_params = out_buffer.GetString();
        return out_params;
    }

    string convertvector(vector<subRes> result){
        rapidjson::Document document;
        document.SetArray();
        rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
        for (const auto sub : result) {
           rapidjson::Value obj(rapidjson::kObjectType);
           obj.AddMember("s", sub.s, allocator);
           obj.AddMember("e", sub.e, allocator);
           obj.AddMember("t", sub.t.c_str(), allocator);
           document.PushBack(obj, allocator);
        }
        rapidjson::StringBuffer strbuf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
        document.Accept(writer);
        return strbuf.GetString();
    }

    string convertvectorV2(vector<subRes> result){
        rapidjson::Document document;
        document.SetObject();
        rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
        rapidjson::Value ObjectArray(rapidjson::kArrayType);
        for (const auto sub : result) {
           rapidjson::Value obj(rapidjson::kObjectType);
           obj.AddMember("s", sub.s, allocator);
           obj.AddMember("e", sub.e, allocator);
           obj.AddMember("t", sub.t.c_str(), allocator);
           ObjectArray.PushBack(obj, allocator);
        }
        document.AddMember("AllTrans", ObjectArray, allocator);
        rapidjson::StringBuffer strbuf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
        document.Accept(writer);
        return strbuf.GetString();
    }
 
    virtual void audiorecognition(google::protobuf::RpcController* cntl_base,
                      const AudioRequest* request,
                      AudioResponse* response,
                      google::protobuf::Closure* done) {
        // This object helps you to call done->Run() in RAII style. If you need
        // to process the request asynchronously, pass done_guard.release().
        baidu::rpc::ClosureGuard done_guard(done); 
       

        string decode_audio_buffer;
        base::Base64Decode(request->audio(), &decode_audio_buffer);
        vector<uint8_t> vec;
        vec.assign(decode_audio_buffer.begin(), decode_audio_buffer.end());
        
        
        AudioFile<float> a;
        bool res = a.loadFromMemory(vec);
        Audio audi(0);
        audi.loadwavfrommem(a);
        audi.split();

        vector<float> buff;
        int len = 0;
        int flag = 1;
        vector<subRes> results;
        int tmp_len = 0;
        while (audi.fetch(buff, len, flag) > 0) {
            int do_idx = rand() % 2; //random number [0,1)
            Accept(buff, do_idx);
            std::string subtxt = GetResult(do_idx);
            Reset(do_idx);
            buff.clear();
            int start_time = (int)(tmp_len/16000.0 * 1000);
            int end_time = (int)((tmp_len + len)/16000.0 * 1000);
            struct subRes subres = {
                start_time,
                end_time,
                subtxt.c_str(),
            };
            tmp_len += len;
            results.push_back(subres);
            com_writelog(COMLOG_NOTICE, "using process: %d, start: %d, end: %d, result: %s", do_idx, start_time, end_time, subtxt.c_str());
        }
        
        // vector<float> inputAudio;
        // for (int i = 0; i < a.getNumSamplesPerChannel(); i++)
        // {
        //     float tempval = 0.0;
        //     for (int channel = 0; channel < a.getNumChannels(); channel++)
        //     {
        //          tempval += a.samples[channel][i] * 32768;
        //     }
        //     inputAudio.emplace_back(tempval);
        // }
        
        // int do_idx = rand() % 2; //random number [0,1)
        // Accept(inputAudio, do_idx);
        // std::string result = GetResult(do_idx);
        // Reset(do_idx);
        // com_writelog(COMLOG_NOTICE, "using process: %d, result: %s", do_idx, result.c_str());
        // std::cout << "Result: " << result << std::endl;

        response->set_err_no(0);
        response->set_err_msg("");
        string jsonresult = convertvector(results);
        response->set_result(jsonresult);
        response->set_cost_time(0);
        results.clear();
    }
};
}  // namespace example



int main(int argc, char* argv[]) {
    // Parse gflags. We recommend you to use gflags as well.
    google::ParseCommandLineFlags(&argc, &argv, true);
    bool flag_auth = false;
    // Setup for `GianoAuthenticator'.
    std::unique_ptr<baidu::rpc::policy::GianoAuthenticator> auth;
    if (flag_auth) {
        if (baas::BAAS_Init() != 0) {
            LOG(ERROR) << "Fail to init BAAS";
            return -1;
        }
        baas::CredentialVerifier 
                ver = baas::ServerUtility::CreateCredentialVerifier();
        auth.reset(new baidu::rpc::policy::GianoAuthenticator(NULL, &ver));
    }

    int ret = com_loadlog("./", "asr.conf");    
    if (ret != 0)
    {
        fprintf(stderr, "load err\n");
        return -1;
    }
    // 打印日志，线程安全
    com_writelog(COMLOG_NOTICE, "server start1"); 


    // Generally you only need one Server.
    baidu::rpc::Server server;
    

    // Instance of your service.
    example::AudioServiceImpl audio_service_impl;

    // Add the service into server. Notice the second parameter, because the
    // service is put on stack, we don't want server to delete it, otherwise
    // use baidu::rpc::SERVER_OWNS_SERVICE.
    if (server.AddService(&audio_service_impl, 
                          baidu::rpc::SERVER_DOESNT_OWN_SERVICE,
                          "/v1/audiorecognition => audiorecognition") != 0) {
        LOG(ERROR) << "Fail to add service";
        return -1;
    }
    InitRecognizer("model.onnx", "words.txt");
    for (int i =0 ;i<= 1; i++){
        int idx = AddRecognizerInstance(); 
    }
    
    srand((unsigned)time(NULL));
    // Start the server.
    baidu::rpc::ServerOptions options;
    options.idle_timeout_sec = -1;
    options.auth = auth.get();
    options.max_concurrency = FLAGS_max_concurrency;
    if (server.Start(FLAGS_port, &options) != 0) {
        LOG(ERROR) << "Fail to start EchoServer";
        return -1;
    }

    // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
    server.RunUntilAskedToQuit();
    return 0;
}