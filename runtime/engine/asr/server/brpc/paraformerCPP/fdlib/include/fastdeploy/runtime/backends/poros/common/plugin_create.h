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

#pragma once

#include <string>
#include <unordered_map>

namespace baidu {
namespace mirana {
namespace poros {

class IPlugin {
 public:
  virtual ~IPlugin() {}
  virtual const std::string who_am_i() = 0;
};

typedef IPlugin* (*plugin_creator_t)();
typedef std::unordered_map<std::string, plugin_creator_t> plugin_creator_map_t;

IPlugin* create_plugin(const std::string& plugin_name);
IPlugin* create_plugin(const std::string& plugin_name,
                       const plugin_creator_map_t& plugin_creator_map);

void create_all_plugins(const plugin_creator_map_t& plugin_creator_map,
                        std::unordered_map<std::string, IPlugin*>& plugin_m);
// void create_all_plugins(std::unordered_map<std::string, IPlugin*>& plugin_m);

template <typename PluginType> IPlugin* default_plugin_creator() {
  return new (std::nothrow) PluginType;
}

void register_plugin_creator(const std::string& plugin_name,
                             plugin_creator_t creator);
void register_plugin_creator(const std::string& plugin_name,
                             plugin_creator_t creator,
                             plugin_creator_map_t& plugin_creator_map);

template <typename PluginType>
void register_plugin_class(const std::string& plugin_name) {
  return register_plugin_creator(plugin_name,
                                 default_plugin_creator<PluginType>);
}

// This version is recommended
template <typename PluginType>
void register_plugin_class(const std::string& plugin_name,
                           plugin_creator_map_t& plugin_creator_map) {
  return register_plugin_creator(
      plugin_name, default_plugin_creator<PluginType>, plugin_creator_map);
}

}  // namespace poros
}  // namespace mirana
}  // namespace baidu

/* vim: set ts=4 sw=4 sts=4 tw=100 */
