/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "node_translator.h"

namespace tvm {
namespace script {
namespace printer {

ObjectGenericFunction<Doc, NodeTranslator>& NodeFragmentTranslators() {
  static ObjectGenericFunction<Doc, NodeTranslator> f;
  return f;
}

ObjectGenericFunction<TypeDoc, NodeTranslator>& VariableTypeTranslators() {
  static ObjectGenericFunction<TypeDoc, NodeTranslator> f;
  return f;
}

TypeDoc NodeTranslator::ToVariableTypeDoc(const ObjectRef& ref) {
  static ObjectGenericFunction<TypeDoc, NodeTranslator>& f = VariableTypeTranslators();
  BeginTranslation(ref);
  TypeDoc result = f(ref, *this);
  EndTranslation();
  return result;
}

bool NodeTranslator::IsLastChild() const {
  auto self = get();
  return self->element_indices.back() + 1 == self->element_counts.back();
}

void NodeTranslator::BeginTranslation(const ObjectRef& ref, size_t element_index,
                                      size_t element_count) {
  get()->path_to_current_node.push_back(ref);
  get()->element_indices.push_back(element_index);
  get()->element_counts.push_back(element_count);
}

void NodeTranslator::EndTranslation() {
  get()->element_counts.pop_back();
  get()->element_indices.pop_back();
  get()->path_to_current_node.pop_back();
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
