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
/*!
 * \brief Translator class to convert IR nodes to Doc as the first step of
 * printing IR fragment into TVMScript
 */
#ifndef TVM_SCRIPT_PRINTER_NODE_TRANSLATOR_H_
#define TVM_SCRIPT_PRINTER_NODE_TRANSLATOR_H_

#include "context.h"
#include "doc.h"
#include "generic_function.h"

namespace tvm {
namespace script {
namespace printer {

class NodeTranslator;

ObjectGenericFunction<Doc, NodeTranslator>& NodeFragmentTranslators();
ObjectGenericFunction<TypeDoc, NodeTranslator>& VariableTypeTranslators();

#define TVMSCRIPT_PRINTER_NODE_TRANSLATOR(DocTranslator)                                \
  TVM_STATIC_REGISTER_GENERIC_FUNCTION(::tvm::script::printer::NodeFragmentTranslators, \
                                       DocTranslator)

#define TVMSCRIPT_PRINTER_VAR_TYPE_DOC_TRANSLATOR(VarTypeDocTranslator)                 \
  TVM_STATIC_REGISTER_GENERIC_FUNCTION(::tvm::script::printer::VariableTypeTranslators, \
                                       VarTypeDocTranslator)

class NodeTranslatorNode : public Object {
 public:
  TranslatorContext context;

  Array<ObjectRef> path_to_current_node;
  // element_indices/counts are associated with the node in `path_to_current_node` with
  // the same index. These represent their order among siblings.
  std::vector<size_t> element_indices;
  std::vector<size_t> element_counts;

  NodeTranslatorNode() = default;
  virtual ~NodeTranslatorNode() = default;

  static constexpr const char* _type_key = "script.printer.NodeTranslator";
  TVM_DECLARE_BASE_OBJECT_INFO(NodeTranslatorNode, Object);
};

class NodeTranslator : public ObjectRef {
 public:
  NodeTranslator() : NodeTranslator(make_object<NodeTranslatorNode>()) {}

  template <typename T, typename = std::enable_if_t<std::is_base_of<Doc, T>::value>>
  T ToDoc(const ObjectRef& ref);

  template <typename DocType, typename NodeType>
  Array<DocType> ToDocArray(const Array<NodeType>& refs);

  ExprDoc ToExprDoc(const ObjectRef& ref) { return ToDoc<ExprDoc>(ref); }

  template <typename NodeType>
  Array<ExprDoc> ToExprDocArray(const Array<NodeType>& refs) {
    return ToDocArray<ExprDoc>(refs);
  }

  TypeDoc ToVariableTypeDoc(const ObjectRef& ref);

  bool IsLastChild() const;

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(NodeTranslator, ObjectRef, NodeTranslatorNode);

 private:
  void BeginTranslation(const ObjectRef& ref, size_t element_index = 0, size_t element_count = 1);
  void EndTranslation();
};

template <typename T, typename>
T NodeTranslator::ToDoc(const ObjectRef& ref) {
  static const ObjectGenericFunction<Doc, NodeTranslator>& f = NodeFragmentTranslators();

  BeginTranslation(ref);
  Doc doc = f(ref, *this);
  doc->origin_ir_node = ref;
  EndTranslation();

  return Downcast<T>(doc);
}

template <typename DocType, typename NodeType>
Array<DocType> NodeTranslator::ToDocArray(const Array<NodeType>& refs) {
  static const ObjectGenericFunction<Doc, NodeTranslator>& f = NodeFragmentTranslators();
  Array<DocType> result;

  size_t index = 0;
  size_t count = refs.size();

  for (auto& ref : refs) {
    BeginTranslation(ref, index, count);

    DocType doc = Downcast<DocType>(f(ref, *this));
    doc->origin_ir_node = ref;
    result.push_back(doc);

    index++;
    EndTranslation();
  }

  return result;
}

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_NODE_TRANSLATOR_H_
