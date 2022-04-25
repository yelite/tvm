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
  TDoc AsDoc(const ObjectRef& obj) const {
    return Downcast<TDoc>(AsDocImpl(obj));
  }

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

 private:
  Doc AsDocImpl(const ObjectRef& obj) const;
};

class IRDocsifier : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IRDocsifier, ObjectRef, IRDocsifierNode);

 public:
  using FType = ObjectFunctor<printer::Doc(const ObjectRef&, IRDocsifier)>;
  TVM_DLL static FType& vtable();
};

inline Doc IRDocsifierNode::AsDocImpl(const ObjectRef& obj) const {
  return IRDocsifier::vtable()(dispatch_tokens.back(), obj, GetRef<IRDocsifier>(this));
}

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_IR_DOCSIFIER_H_
