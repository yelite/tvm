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
#include "./traced_object.h"

namespace tvm {

class ObjectPath;

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
  /*!
   * \brief Transform the input object into TDoc
   */

  template <class TDoc>
  TDoc AsDoc(const TracedObject<ObjectRef>& obj) const {
    auto result = Downcast<TDoc>(AsDocImpl(obj));
    result->source = obj.Get();
    result->paths.push_back(obj.GetPath());
    return result;
  }

  ExprDoc AsExprDoc(const TracedObject<ObjectRef>& ref) { return AsDoc<ExprDoc>(ref); }

  /*!
   * \brief Push a new dispatch token into the stack
   * \details The top dispatch token decides which dispatch table to use
   *          when printing Object. This method returns a RAII guard which
   *          pops the token when going out of the scope.
   */
  WithCtx WithDispatchToken(const String& token) {
    this->dispatch_tokens.push_back(token);
    return WithCtx(nullptr, [this]() { this->dispatch_tokens.pop_back(); });
  }

  /*!
   * \brief Push a new frame the stack
   * \details Frame contains the contextual information that's needed during printing,
   *          for example, variables in the scope. This method returns a RAII guard which
   *          pops the frame and call the cleanup method of frame when going out of the scope.
   */
  WithCtx WithFrame(const Frame& frame) {
    frame->EnterWithScope();
    this->frames.push_back(frame);
    return WithCtx(nullptr, [this]() {
      Frame frame = this->frames.back();
      this->frames.pop_back();
      frame->ExitWithScope();
    });
  }

  /*!
   * \brief Get the top frame with type FrameType
   */
  template <typename FrameType>
  Optional<FrameType> GetFrame() const;

  /*!
   * Get an array of frames of `FrameType`. The first element in the array is the latest frame
   * added.
   */
  template <typename FrameType>
  Array<FrameType> GetFrames() const;

 private:
  Doc AsDocImpl(const TracedObject<ObjectRef>& obj) const;
};

class IRDocsifier : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IRDocsifier, ObjectRef, IRDocsifierNode);

 public:
  IRDocsifier(Map<String, String> ir_prefix);

  using FType = TracedObjectFunctor<printer::Doc, IRDocsifier>;
  TVM_DLL static FType& vtable();
};

template <typename FrameType>
Optional<FrameType> IRDocsifierNode::GetFrame() const {
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    if (const auto* f = (*it).as<typename FrameType::ContainerType>()) {
      return GetRef<FrameType>(f);
    }
  }
  return NullOpt;
}

template <typename FrameType>
Array<FrameType> IRDocsifierNode::GetFrames() const {
  Array<FrameType> result;
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    if (const auto* f = (*it).as<typename FrameType::ContainerType>()) {
      result.push_back(GetRef<FrameType>(f));
    }
  }
  return result;
}

inline Doc IRDocsifierNode::AsDocImpl(const TracedObject<ObjectRef>& obj) const {
  return IRDocsifier::vtable()(dispatch_tokens.back(), obj.Get(), obj.GetPath(),
                               GetRef<IRDocsifier>(this));
}

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_IR_DOCSIFIER_H_
