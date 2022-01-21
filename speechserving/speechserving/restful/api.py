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
from fastapi import APIRouter

router = APIRouter()


router.include_router(auth_router)
router.include_router(user_router)
router.include_router(profile_router)
router.include_router(comment_router)
router.include_router(article_router)
router.include_router(tag_router)




def init_app(app):
    
    app.include_router(router)
