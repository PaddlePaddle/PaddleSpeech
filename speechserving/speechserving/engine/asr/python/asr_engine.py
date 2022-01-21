# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from engine import BaseEngine

__all__ = ['ASREngine']

class ASREngine(BaseEngine):

    def __init__(self, name):
        super(ASREngine, self).__init__()


    def init(self):
        pass

    def postprocess(self):
        pass

    def run(self):
        pass



if __name__ == "__main__":
    # test Singleton 
    class1 = ASREngine("ASREngine")
    class2 = ASREngine()
    print(class1 is class2)
    print(id(class1))
    print(id(class2))

