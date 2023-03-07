// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void* PPSHandle_t;

typedef enum {
    PPS_VAD_ILLEGAL = 0,  // error
    PPS_VAD_SIL,          // silence
    PPS_VAD_START,        // start speech
    PPS_VAD_SPEECH,       // in speech
    PPS_VAD_END,          // end speech
    PPS_VAD_NUMSTATES,    // number of states
} PPSVadState_t;

PPSHandle_t PPSVadCreateInstance(const char* conf_path);

int PPSVadDestroyInstance(PPSHandle_t instance);

int PPSVadReset(PPSHandle_t instance);

int PPSVadChunkSizeSamples(PPSHandle_t instance);

PPSVadState_t PPSVadFeedForward(PPSHandle_t instance,
                                float* chunk,
                                int num_element);

#ifdef __cplusplus
}
#endif  // __cplusplus