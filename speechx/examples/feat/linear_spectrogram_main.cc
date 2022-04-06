// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

// todo refactor, repalce with gtest

#include "base/flags.h"
#include "base/log.h"
#include "kaldi/feat/wave-reader.h"
#include "kaldi/util/kaldi-io.h"
#include "kaldi/util/table-types.h"

#include "frontend/audio/audio_cache.h"
#include "frontend/audio/data_cache.h"
#include "frontend/audio/feature_cache.h"
#include "frontend/audio/frontend_itf.h"
#include "frontend/audio/linear_spectrogram.h"
#include "frontend/audio/normalizer.h"

DEFINE_string(wav_rspecifier, "", "test wav scp path");
DEFINE_string(feature_wspecifier, "", "output feats wspecifier");
DEFINE_string(cmvn_write_path, "./cmvn.ark", "write cmvn");


std::vector<float> mean_{
    -13730251.531853663, -12982852.199316509, -13673844.299583456,
    -13089406.559646806, -12673095.524938712, -12823859.223276224,
    -13590267.158903603, -14257618.467152044, -14374605.116185192,
    -14490009.21822485,  -14849827.158924166, -15354435.470563512,
    -15834149.206532761, -16172971.985514281, -16348740.496746974,
    -16423536.699409386, -16556246.263649225, -16744088.772748645,
    -16916184.08510357,  -17054034.840031497, -17165612.509455364,
    -17255955.470915023, -17322572.527648456, -17408943.862033736,
    -17521554.799865916, -17620623.254924215, -17699792.395918526,
    -17723364.411134344, -17741483.4433254,   -17747426.888704527,
    -17733315.928209435, -17748780.160905756, -17808336.883775543,
    -17895918.671983004, -18009812.59173023,  -18098188.66548325,
    -18195798.958462656, -18293617.62980999,  -18397432.92077201,
    -18505834.787318766, -18585451.8100908,   -18652438.235649142,
    -18700960.306275308, -18734944.58792185,  -18737426.313365128,
    -18735347.165987637, -18738813.444170244, -18737086.848890636,
    -18731576.2474336,   -18717405.44095871,  -18703089.25545657,
    -18691014.546456724, -18692460.568905357, -18702119.628629155,
    -18727710.621126678, -18761582.72034647,  -18806745.835547544,
    -18850674.8692112,   -18884431.510951452, -18919999.992506847,
    -18939303.799078144, -18952946.273760635, -18980289.22996379,
    -19011610.17803294,  -19040948.61805145,  -19061021.429847397,
    -19112055.53768819,  -19149667.414264943, -19201127.05091321,
    -19270250.82564605,  -19334606.883057203, -19390513.336589377,
    -19444176.259208687, -19502755.000038862, -19544333.014549147,
    -19612668.183176614, -19681902.19006569,  -19771969.951249883,
    -19873329.723376893, -19996752.59235844,  -20110031.131400537,
    -20231658.612529557, -20319378.894054495, -20378534.45718066,
    -20413332.089584175, -20438147.844177883, -20443710.248040095,
    -20465457.02238927,  -20488610.969337028, -20516295.16424432,
    -20541423.795738827, -20553192.874953747, -20573605.50701977,
    -20577871.61936797,  -20571807.008916274, -20556242.38912231,
    -20542199.30819195,  -20521239.063551214, -20519150.80004532,
    -20527204.80248933,  -20536933.769257784, -20543470.522332076,
    -20549700.089992985, -20551525.24958494,  -20554873.406493705,
    -20564277.65794227,  -20572211.740052115, -20574305.69550465,
    -20575494.450104576, -20567092.577932164, -20549302.929608088,
    -20545445.11878376,  -20546625.326603737, -20549190.03499401,
    -20554824.947828256, -20568341.378989458, -20577582.331383612,
    -20577980.519402675, -20566603.03458152,  -20560131.592262644,
    -20552166.469060015, -20549063.06763577,  -20544490.562339947,
    -20539817.82346569,  -20528747.715731595, -20518026.24576161,
    -20510977.844974525, -20506874.36087992,  -20506731.11977665,
    -20510482.133420516, -20507760.92101862,  -20494644.834457114,
    -20480107.89304893,  -20461312.091867123, -20442941.75080173,
    -20426123.02834838,  -20424607.675283,    -20426810.369107097,
    -20434024.50097819,  -20437404.75544205,  -20447688.63916367,
    -20460893.335563846, -20482922.735127095, -20503610.119434915,
    -20527062.76448319,  -20557830.035128627, -20593274.72068722,
    -20632528.452965066, -20673637.471334763, -20733106.97143075,
    -20842921.0447562,   -21054357.83621519,  -21416569.534189366,
    -21978460.272811692, -22753170.052172784, -23671344.10563395,
    -24613499.293358143, -25406477.12230188,  -25884377.82156489,
    -26049040.62791664,  -26996879.104431007};
std::vector<float> variance_{
    213747175.10846674, 188395815.34302503, 212706429.10966414,
    199109025.81461075, 189235901.23864496, 194901336.53253657,
    217481594.29306737, 238689869.12327808, 243977501.24115244,
    248479623.6431067,  259766741.47116545, 275516766.7790273,
    291271202.3691234,  302693239.8220509,  308627358.3997694,
    311143911.38788426, 315446105.07731867, 321705430.9341829,
    327458907.4659941,  332245072.43223983, 336251717.5935284,
    339694069.7639722,  342188204.4322228,  345587110.31313115,
    349903086.2875232,  353660214.20643026, 356700344.5270885,
    357665362.3529641,  358493352.05658793, 358857951.620328,
    358375239.52774596, 358899733.6342954,  361051818.3511561,
    364361716.05025816, 368750322.3771452,  372047800.6462831,
    375655861.1349018,  379358519.1980013,  383327605.3935181,
    387458599.282341,   390434692.3406868,  392994486.35057056,
    394874418.04603153, 396230525.79763395, 396365592.0414835,
    396334819.8242737,  396488353.19250053, 396438877.00744957,
    396197980.4459586,  395590921.6672991,  395001107.62072515,
    394528291.7318225,  394593110.424006,   395018405.59353715,
    396110577.5415993,  397506704.0371068,  399400197.4657644,
    401243568.2468382,  402687134.7805103,  404136047.2872507,
    404883170.001883,   405522253.219517,   406660365.3626476,
    407919346.0991902,  409045348.5384909,  409759588.7889818,
    411974821.8564483,  413489718.78201455, 415535392.56684107,
    418466481.97674364, 421104678.35678065, 423405392.5200779,
    425550570.40798235, 427929423.9579701,  429585274.253478,
    432368493.55181056, 435193587.13513297, 438886855.20476013,
    443058876.8633751,  448181232.5093362,  452883835.6332396,
    458056721.77926534, 461816531.22735566, 464363620.1970998,
    465886343.5057493,  466928872.0651,     467180536.42647296,
    468111848.70714295, 469138695.3071312,  470378429.6930793,
    471517958.7132626,  472109050.4262365,  473087417.0177867,
    473381322.04648733, 473220195.85483915, 472666071.8998819,
    472124669.87879956, 471298571.411737,   471251033.2902761,
    471672676.43128747, 472177147.2193172,  472572361.7711908,
    472968783.7751127,  473156295.4164052,  473398034.82676554,
    473897703.5203811,  474328271.33112127, 474452670.98002136,
    474549003.99284613, 474252887.13567275, 473557462.909069,
    473483385.85193115, 473609738.04855174, 473746944.82085115,
    474016729.91696435, 474617321.94138587, 475045097.237122,
    475125402.586558,   474664112.9824912,  474426247.5800283,
    474104075.42796475, 473978219.7273978,  473773171.7798875,
    473578534.69508696, 473102924.16904145, 472651240.5232615,
    472374383.1810912,  472209479.6956096,  472202298.8921673,
    472370090.76781124, 472220933.99374026, 471625467.37106377,
    470994646.51883453, 470182428.9637543,  469348211.5939578,
    468570387.4467277,  468540442.7225135,  468672018.90414184,
    468994346.9533251,  469138757.58201426, 469553915.95710236,
    470134523.38582784, 471082421.62055486, 471962316.51804745,
    472939745.1708408,  474250621.5944825,  475773933.43199486,
    477465399.71087736, 479218782.61382693, 481752299.7930922,
    486608947.8984568,  496119403.2067917,  512730085.5704984,
    539048915.2641417,  576285298.3548826,  621610270.2240586,
    669308196.4436442,  710656993.5957186,  736344437.3725077,
    745481288.0241544,  801121432.9925804};
int count_ = 912592;

void WriteMatrix() {
    kaldi::Matrix<double> cmvn_stats(2, mean_.size() + 1);
    for (size_t idx = 0; idx < mean_.size(); ++idx) {
        cmvn_stats(0, idx) = mean_[idx];
        cmvn_stats(1, idx) = variance_[idx];
    }
    cmvn_stats(0, mean_.size()) = count_;
    kaldi::WriteKaldiObject(cmvn_stats, FLAGS_cmvn_write_path, false);
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);

    kaldi::SequentialTableReader<kaldi::WaveHolder> wav_reader(
        FLAGS_wav_rspecifier);
    kaldi::BaseFloatMatrixWriter feat_writer(FLAGS_feature_wspecifier);
    WriteMatrix();


    int32 num_done = 0, num_err = 0;

    // feature pipeline: wave cache --> decibel_normalizer --> hanning
    // window -->linear_spectrogram --> global cmvn -> feat cache

    // std::unique_ptr<ppspeech::FrontendInterface> data_source(new
    // ppspeech::DataCache());
    std::unique_ptr<ppspeech::FrontendInterface> data_source(
        new ppspeech::AudioCache());

    ppspeech::DecibelNormalizerOptions db_norm_opt;
    std::unique_ptr<ppspeech::FrontendInterface> db_norm(
        new ppspeech::DecibelNormalizer(db_norm_opt, std::move(data_source)));

    ppspeech::LinearSpectrogramOptions opt;
    opt.frame_opts.frame_length_ms = 20;
    opt.frame_opts.frame_shift_ms = 10;
    opt.frame_opts.dither = 0.0;
    opt.frame_opts.remove_dc_offset = false;
    opt.frame_opts.window_type = "hanning";
    opt.frame_opts.preemph_coeff = 0.0;
    LOG(INFO) << "frame length (ms): " << opt.frame_opts.frame_length_ms;
    LOG(INFO) << "frame shift (ms): " << opt.frame_opts.frame_shift_ms;

    std::unique_ptr<ppspeech::FrontendInterface> linear_spectrogram(
        new ppspeech::LinearSpectrogram(opt, std::move(db_norm)));

    std::unique_ptr<ppspeech::FrontendInterface> cmvn(new ppspeech::CMVN(
        FLAGS_cmvn_write_path, std::move(linear_spectrogram)));

    ppspeech::FeatureCache feature_cache(kint16max, std::move(cmvn));
    LOG(INFO) << "feat dim: " << feature_cache.Dim();

    int sample_rate = 16000;
    float streaming_chunk = 0.36;
    int chunk_sample_size = streaming_chunk * sample_rate;
    LOG(INFO) << "sr: " << sample_rate;
    LOG(INFO) << "chunk size (s): " << streaming_chunk;
    LOG(INFO) << "chunk size (sample): " << chunk_sample_size;


    for (; !wav_reader.Done(); wav_reader.Next()) {
        std::string utt = wav_reader.Key();
        const kaldi::WaveData& wave_data = wav_reader.Value();
        LOG(INFO) << "process utt: " << utt;

        int32 this_channel = 0;
        kaldi::SubVector<kaldi::BaseFloat> waveform(wave_data.Data(),
                                                    this_channel);
        int tot_samples = waveform.Dim();
        LOG(INFO) << "wav len (sample): " << tot_samples;

        int sample_offset = 0;
        std::vector<kaldi::Vector<BaseFloat>> feats;
        int feature_rows = 0;
        while (sample_offset < tot_samples) {
            int cur_chunk_size =
                std::min(chunk_sample_size, tot_samples - sample_offset);

            kaldi::Vector<kaldi::BaseFloat> wav_chunk(cur_chunk_size);
            for (int i = 0; i < cur_chunk_size; ++i) {
                wav_chunk(i) = waveform(sample_offset + i);
            }

            kaldi::Vector<BaseFloat> features;
            feature_cache.Accept(wav_chunk);
            if (cur_chunk_size < chunk_sample_size) {
                feature_cache.SetFinished();
            }
            feature_cache.Read(&features);
            if (features.Dim() == 0) break;

            feats.push_back(features);
            sample_offset += cur_chunk_size;
            feature_rows += features.Dim() / feature_cache.Dim();
        }

        int cur_idx = 0;
        kaldi::Matrix<kaldi::BaseFloat> features(feature_rows,
                                                 feature_cache.Dim());
        for (auto feat : feats) {
            int num_rows = feat.Dim() / feature_cache.Dim();
            for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
                for (size_t col_idx = 0; col_idx < feature_cache.Dim();
                     ++col_idx) {
                    features(cur_idx, col_idx) =
                        feat(row_idx * feature_cache.Dim() + col_idx);
                }
                ++cur_idx;
            }
        }
        feat_writer.Write(utt, features);

        if (num_done % 50 == 0 && num_done != 0)
            KALDI_VLOG(2) << "Processed " << num_done << " utterances";
        num_done++;
    }
    KALDI_LOG << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
}
