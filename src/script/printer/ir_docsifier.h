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
#ifndef TVM_SCRIPT_PRINTER_IR_DOCSIFIER_H_
#define TVM_SCRIPT_PRINTER_IR_DOCSIFIER_H_

#include <tvm/node/node.h>
#include <tvm/support/with.h>

#include "./doc.h"
#include "./frame.h"
#include "./functor.h"
#include "./symbol_table.h"

namespace tvm {
namespace script {
namespace printer {

using WithCtx = With<ContextManager>;

class IRDocsifierNode : public Object {
 public:
  Map<String, String> ir_prefix;
  SymbolTable sym;
  Array<Frame> frames;
  Array<String> dispatch_tokens;

  Array<ObjectRef> path_to_current_node;
  // element_indices/counts are associated with the node in `path_to_current_node` with
  // the same index. These represent their order among siblings.
  std::vector<size_t> node_indices;
  std::vector<size_t> node_counts;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("ir_prefix", &ir_prefix);
    v->Visit("sym", &sym);
    v->Visit("frames", &frames);
    v->Visit("dispatch_tokens", &dispatch_tokens);
  }

  static constexpr const char* _type_key = "script.IRDocsifier";
  TVM_DECLARE_FINAL_OBJECT_INFO(IRDocsifierNode, Object);

 public:
  template <class TDoc>
  TDoc AsDoc(const ObjectRef& obj) {
    return Downcast<TDoc>(AsDocImpl(obj));
  }

  template <typename DocType, typename NodeType>
  Array<DocType> AsDocArray(const Array<NodeType>& refs);

  ExprDoc AsExprDoc(const ObjectRef& ref) { return AsDoc<ExprDoc>(ref); }

  template <typename NodeType>
  Array<ExprDoc> AsExprDocArray(const Array<NodeType>& refs) {
    return AsDocArray<ExprDoc>(refs);
  }

  ExprDoc ToVariableTypeDoc(const ObjectRef& ref);

  WithCtx WithDispatchToken(const String& token) {
    this->dispatch_tokens.push_back(token);
    return WithCtx(nullptr, [this]() { this->dispatch_tokens.pop_back(); });
  }

  WithCtx WithFrame(const Frame& frame) {
    frame->EnterWithScope();
    this->frames.push_back(frame);
    return WithCtx(nullptr, [this]() {
      Frame frame = this->frames.back();
      this->frames.pop_back();
      frame->ExitWithScope();
    });
  }

  template <typename FrameType>
  Optional<FrameType> GetFrame() const;

  bool IsLastChild() const { return node_indices.back() + 1 == node_counts.back(); };

 private:
  Doc AsDocImpl(const ObjectRef& obj, size_t node_index = 0, size_t node_count = 1);
};

class IRDocsifier : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IRDocsifier, ObjectRef, IRDocsifierNode);

 public:
  IRDocsifier(Map<String, String> ir_prefix);

  using FType = ObjectFunctor<printer::Doc(const ObjectRef&, IRDocsifier)>;
  TVM_DLL static FType& vtable();
  TVM_DLL static FType& var_type_vtable();
};

template <typename FrameType>
Optional<FrameType> IRDocsifierNode::GetFrame() const {
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    if ((*it)->IsInstance<FrameType>()) {
      return *it;
    }
  }
  return NullOpt;
}

inline Doc ToDocWithFunctor(IRDocsifierNode* ir_docsifier, IRDocsifier::FType& functor,
                            const ObjectRef& obj, size_t node_index, size_t node_count) {
  ir_docsifier->path_to_current_node.push_back(obj);
  ir_docsifier->node_indices.push_back(node_index);
  ir_docsifier->node_counts.push_back(node_count);

  Doc doc = functor(ir_docsifier->dispatch_tokens.back(), obj, GetRef<IRDocsifier>(ir_docsifier));
  doc->source = obj;

  ir_docsifier->node_counts.pop_back();
  ir_docsifier->node_indices.pop_back();
  ir_docsifier->path_to_current_node.pop_back();

  return doc;
}

inline Doc IRDocsifierNode::AsDocImpl(const ObjectRef& obj, size_t node_index, size_t node_count) {
  return ToDocWithFunctor(this, IRDocsifier::vtable(), obj, node_index, node_count);
}

inline ExprDoc IRDocsifierNode::ToVariableTypeDoc(const ObjectRef& obj) {
  // TODO: remove downcase and make functor return ExprDoc directly
  return Downcast<ExprDoc>(ToDocWithFunctor(this, IRDocsifier::var_type_vtable(), obj, 0, 1));
}

template <typename DocType, typename NodeType>
Array<DocType> IRDocsifierNode::AsDocArray(const Array<NodeType>& refs) {
  Array<DocType> result;

  size_t index = 0;
  size_t count = refs.size();

  for (auto& ref : refs) {
    result.push_back(Downcast<DocType>(AsDocImpl(ref, index, count)));
    index++;
  }

  return result;
}

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_IR_DOCSIFIER_H_
