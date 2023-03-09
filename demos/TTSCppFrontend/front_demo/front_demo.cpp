#include <string>
//#include "utils/dir_utils.h"
#include "front/front_interface.h"
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <map>

DEFINE_string(sentence, "你好，欢迎使用语音合成服务", "Text to be synthesized");
DEFINE_string(front_conf, "./front_demo/front.conf", "Front conf file");
//DEFINE_string(seperate_tone, "true", "If true, get phoneids and tonesid");


int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // 实例化文本前端引擎
    speechnn::FrontEngineInterface *front_inst = nullptr;
    front_inst = new speechnn::FrontEngineInterface(FLAGS_front_conf);
    if ((!front_inst) || (front_inst->init())) {
        LOG(ERROR) << "Creater tts engine failed!";
        if (front_inst != nullptr) {
            delete front_inst;
        }
        front_inst = nullptr;
        return -1;
    }

    std::wstring ws_sentence = speechnn::utf8string2wstring(FLAGS_sentence);

    // 繁体转简体
    std::wstring sentence_simp;
    front_inst->Trand2Simp(ws_sentence, sentence_simp); 
    ws_sentence = sentence_simp;

    std::string s_sentence;
    std::vector<std::wstring> sentence_part;
    std::vector<int> phoneids = {};
    std::vector<int> toneids = {};

    // 根据标点进行分句
    LOG(INFO) << "Start to segment sentences by punctuation";
    front_inst->SplitByPunc(ws_sentence, sentence_part); 
    LOG(INFO) << "Segment sentences through punctuation successfully";

    // 分句后获取音素id
    LOG(INFO) << "Start to get the phoneme and tone id sequence of each sentence";
    for(int i = 0; i < sentence_part.size(); i++) {

        LOG(INFO) << "Raw sentence is: " << speechnn::wstring2utf8string(sentence_part[i]);
        front_inst->SentenceNormalize(sentence_part[i]);
        s_sentence = speechnn::wstring2utf8string(sentence_part[i]);
        LOG(INFO) << "After normalization sentence is: " << s_sentence;
        
        if (0 != front_inst->GetSentenceIds(s_sentence, phoneids, toneids)) {
            LOG(ERROR) << "TTS inst get sentence phoneids and toneids failed";
            return -1;
        }
            
    }
    LOG(INFO) << "The phoneids of the sentence is: " << limonp::Join(phoneids.begin(), phoneids.end(), " ");
    LOG(INFO) << "The toneids of the sentence is: " << limonp::Join(toneids.begin(), toneids.end(), " ");
    LOG(INFO) << "Get the phoneme id sequence of each sentence successfully";
 
    return EXIT_SUCCESS;
}

