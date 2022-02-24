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

#include "tvmscript_unified_printer.h"

#include <tvm/node/functor.h>

#include <cstdint>
#include <string>

#include "doc.h"
#include "tvm/runtime/container/array.h"
#include "tvm/runtime/container/string.h"
#include "tvm/runtime/object.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/function.h"
#include "tvm/tir/op.h"
#include "tvm/tir/stmt.h"

namespace tvm {

namespace {
using namespace tir;
using runtime::ArrayNode;
}  // namespace

enum class UnifiedExprPrecedence : int {
  /*! \brief Identity(e.g., IntImm, Var) and function call(e.g., floordiv, min) */
  kIdentity = 0,
  /*!
   * \brief Multiplication(*), division(/), and remainder(%)
   * \note floorDiv, floorMod is marked as kIdentity since they are function calls.
   */
  kMultiplicationDivision = 1,
  /*! \brief Addition(+) and subtraction(-) */
  kAdditionSubtraction = 2,
  /*! \brief For relational operators < and <= and > and >= respectively */
  kRelational = 3,
  /*! \brief For equality operators = and != respectively */
  kEquality = 4,
  /*! \brief And(&&) */
  kAnd = 5,
  /*! \brief Or(||) */
  kOr = 6,
  /*! \brief Unknown precedence */
  kUnknown = 7,
};

class TVMScriptUnifiedPrinter {
 public:
  explicit TVMScriptUnifiedPrinter(const String& tir_prefix) : tir_prefix_(tir_prefix){};
  using FType = NodeFunctor<Doc(const ObjectRef&, TVMScriptUnifiedPrinter&)>;
  static FType& vtable();

  Doc PrintNode(const ObjectRef& ref);
  Doc PrintExtraVarDeclaration();

  template <typename T>
  Doc PrintConstScalar(DataType dtype, const T* data) const {
    Doc doc;
    std::ostringstream os;
    if (dtype.is_float() || dtype.is_float16() || dtype.is_bfloat16()) {
      os.precision(17);
    }
    os << data[0];
    if (dtype == DataType::Int(32)) {
      doc << Doc::Text(os.str());
    } else if (dtype == DataType::Bool()) {
      doc << Doc::Text(data[0] ? "True" : "False");
    } else {
      doc << tir_prefix_ << "." << runtime::DLDataType2String(dtype) << "(" << Doc::Text(os.str())
          << ")";
    }
    return doc;
  }

  Doc PrintDType(DataType dtype) { return Doc::StrLiteral(runtime::DLDataType2String(dtype)); }
  Doc PrintBufferAnnotation(Buffer buf);

  Doc PrintTuple(const ArrayNode* array);

  using BinOpPrinter = std::function<Doc(Doc&&, Doc&&)>;  // (left, right) -> result
  Doc PrintBinOp(const PrimExpr left, const PrimExpr right, UnifiedExprPrecedence op_precedence,
                 BinOpPrinter bin_op_printer);
  Doc PrintBinOp(const PrimExpr left, const PrimExpr right, UnifiedExprPrecedence op_precedence,
                 const std::string& bin_op);  // Q: pass by value, ref or rvalue ref?

  void SetExprPrecedence(UnifiedExprPrecedence precedence) { last_expr_precedence_ = precedence; };
  Doc WithTirPrefix(const std::string& symbol) {
    Doc doc;
    doc << tir_prefix_ << "." << symbol;
    return doc;
  }

  void onVarUsed(Var var) {
    if (vars_in_scope_.count(var) == 0) {
      undeclared_vars_.insert(var);
    }
  }
  void onVarEnterScope(Var var) { vars_in_scope_.insert(var); };
  void onVarExitScope(Var var) { vars_in_scope_.erase(var); };

  void onBufferUsed(Buffer buf) {
    if (buffer_in_scope_.count(buf) == 0) {
      undeclared_buffers_.insert(buf);
    }
  }
  void onBufferEnterScope(Buffer buf) { buffer_in_scope_.insert(buf); };
  void onBufferExitScope(Buffer buf) { buffer_in_scope_.erase(buf); };

 protected:
  String tir_prefix_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> vars_in_scope_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> undeclared_vars_;
  // Sum type? or just use ObjectRef to unify this with vars_in_scope
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_in_scope_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> undeclared_buffers_;

  // Maybe just split expr and stmt printing into different vtable,
  // and make expr printing return Pair<Doc, Precedence>.
  UnifiedExprPrecedence last_expr_precedence_ = UnifiedExprPrecedence::kUnknown;
};

TVMScriptUnifiedPrinter::FType& TVMScriptUnifiedPrinter::vtable() {
  static FType inst;
  return inst;
}

Doc TVMScriptUnifiedPrinter::PrintBinOp(const PrimExpr lhs, const PrimExpr rhs,
                                        UnifiedExprPrecedence op_precedence,
                                        BinOpPrinter bin_op_printer) {
  Doc lhs_doc = PrintNode(lhs);
  if (last_expr_precedence_ > op_precedence) {
    Doc tmp;
    tmp << "(" << lhs_doc << ")";
    lhs_doc = tmp;
  }
  last_expr_precedence_ = UnifiedExprPrecedence::kUnknown;

  Doc rhs_doc = PrintNode(rhs);
  if (last_expr_precedence_ >= op_precedence) {
    Doc tmp;
    tmp << "(" << rhs_doc << ")";
    lhs_doc = tmp;
  }
  last_expr_precedence_ = UnifiedExprPrecedence::kUnknown;

  Doc result = bin_op_printer(std::move(lhs_doc), std::move(rhs_doc));
  last_expr_precedence_ = op_precedence;
  return result;
}

Doc TVMScriptUnifiedPrinter::PrintBinOp(const PrimExpr left, const PrimExpr right,
                                        UnifiedExprPrecedence op_precedence,
                                        const std::string& bin_op) {
  return PrintBinOp(left, right, op_precedence, [=](Doc&& left, Doc&& right) {
    Doc doc;
    doc << left << bin_op << right;
    return doc;
  });
}

Doc TVMScriptUnifiedPrinter::PrintBufferAnnotation(Buffer buf) {
  std::vector<Doc> type_params;
  std::transform(buf->shape.begin(), buf->shape.end(), std::back_inserter(type_params),
                 [&](auto n) { return PrintNode(n); });
  return PrintNode(buf) << ": " << WithTirPrefix("Buffer") << "["
                        << Doc::Concat(type_params, Doc::StrLiteral(", ")) << ", "
                        << PrintDType(buf->dtype) << "]";
}

Doc TVMScriptUnifiedPrinter::PrintNode(const ObjectRef& ref) { return vtable()(ref, *this); }
Doc TVMScriptUnifiedPrinter::PrintExtraVarDeclaration() {
  Doc doc;
  if (undeclared_vars_.empty() || undeclared_buffers_.empty()) {
    return doc;
  }
  for (const auto& var : undeclared_vars_) {
    doc << var->name_hint << ": " << PrintNode(GetType(var)) << Doc::NewLine();
  }
  for (const auto& buf : undeclared_buffers_) {
      doc << PrintBufferAnnotation(buf) << Doc::NewLine();
  }
  doc << Doc::NewLine();
  return doc;
}

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<PrimFuncNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) {
      Doc doc;
      const auto func = Downcast<PrimFunc>(n);
      std::string func_name = "func";
      const auto& it = func->attrs->dict.find("global_symbol");
      if (it != func->attrs->dict.end()) {
        func_name = Downcast<String>((*it).second);
      }
      doc << "@" << p.WithTirPrefix("prim_func") << Doc::NewLine();
      doc << "def " << func_name << "(";

      std::vector<Doc> params;
      for (const auto& param : func->params) {
        auto it = func->buffer_map.find(param);
        if (it != func->buffer_map.end()) {
          const Buffer& buf = (*it).second;
          p.onBufferEnterScope(buf);
          params.push_back(p.PrintBufferAnnotation(buf));
          continue;
        }
        params.push_back(p.PrintNode(param) << ": " << p.PrintNode(GetType(param)));
      }
      doc << Doc::Concat(params, Doc::Text(", ")) << ") -> " << p.PrintNode(func->ret_type) << ":";

      Doc body;

      // attrs
      if (func->attrs.defined()) {
        body << Doc::NewLine() << "# function attr dict" << Doc::NewLine()
             << p.WithTirPrefix("func_attr") << "({";
        std::vector<Doc> attrs;
        for (const auto& it : func->attrs->dict) {
          attrs.push_back(Doc::StrLiteral(it.first) << ": " << p.PrintNode(it.second));
        }
        body << Doc::Concat(attrs, Doc::Text(", ")) << "})";
      }

      if (func->body->IsInstance<BlockRealizeNode>() &&
          func->body.as<BlockRealizeNode>()->iter_values.empty()) {
        const BlockNode* block = func->body.as<BlockRealizeNode>()->block.get();
        if (block->annotations.empty()) {
          // Skip print root block
          body << Doc::NewLine() << "# with " << p.WithTirPrefix("block") << "(\"root\")"
               << Doc::NewLine();
          body << p.PrintNode(GetRef<ObjectRef>(block));
        } else {
          body << p.PrintNode(func->body);
        }
      } else {
        body << p.PrintNode(func->body);
      }

      for (const auto& param : func->params) {
        auto it = func->buffer_map.find(param);
        if (it != func->buffer_map.end()) {
          p.onBufferExitScope((*it).second);
          continue;
        }
      }

      doc << Doc::Indent(4, body);
      return doc;
    });

Doc PrintBlockVars(const BlockRealize block_realize, TVMScriptUnifiedPrinter& p) {
  Doc doc;
  const auto block = block_realize->block;
  ICHECK_EQ(block->iter_vars.size(), block_realize->iter_values.size());

  // TODO: handle remap

  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& value = block_realize->iter_values[i];
    p.onVarEnterScope(iter_var->var);
    doc << Doc::NewLine() << p.PrintNode(iter_var->var) << " = " << p.WithTirPrefix("axis");
    switch (iter_var->iter_type) {
      case kDataPar:
        doc << ".spatial";
        break;
      case kCommReduce:
        doc << ".reduce";
        break;
      case kOrdered:
        doc << ".scan";
        break;
      case kOpaque:
        doc << ".opaque";
        break;
      default:
        LOG(FATAL) << "Unknown block var iter type: " << iter_var->iter_type;
        break;
    }
    doc << "(";
    const Range& dom = iter_var->dom;
    if (is_zero(dom->min)) {
      doc << p.PrintNode(dom->extent);
    } else {
      doc << "(" << p.PrintNode(dom->min) << ", " << p.PrintNode(dom->min + dom->extent) << ")";
    }
    doc << ", " << p.PrintNode(value) << ")";
  }
  return doc;
}

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<BlockRealizeNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) {
      const auto block_realize = Downcast<BlockRealize>(n);
      const auto block = block_realize->block;
      Doc doc;
      // TODO: optional info
      // print block name and block vars
      doc << "with " << p.WithTirPrefix("block") << "(";
      if (!block->name_hint.empty()) {
        doc << Doc::StrLiteral(block->name_hint);
      }
      doc << "):";
      Doc block_var = PrintBlockVars(block_realize, p);
      // print body
      Doc body = p.PrintNode(block);
      doc << Doc::Indent(4, block_var << Doc::NewLine() << body);
      for (const auto& iter_var : block->iter_vars) {
        p.onVarExitScope(iter_var->var);
      }
      return doc;
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<BlockNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) {
      const auto block = Downcast<Block>(n);
      Doc body;
      // TODO: T.alloc_buffer and match_buffer and init
      body << p.PrintNode(block->body);
      return body;
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<ForNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) {
      const auto for_ref = Downcast<For>(n);
      Doc doc;
      p.onVarEnterScope(for_ref->loop_var);
      doc << "for " << p.PrintNode(for_ref->loop_var) << " in "
          << p.WithTirPrefix(std::string(ForKind2String(for_ref->kind))) << "(";
      if (is_zero(for_ref->min)) {
        doc << p.PrintNode(for_ref->extent);
      } else {
        doc << p.PrintNode(for_ref->min) << ", " << p.PrintNode(for_ref->min + for_ref->extent);
      }
      // TODO: annotation, thread binding
      doc << "):";
      doc << Doc::Indent(4, Doc::NewLine() << p.PrintNode(for_ref->body));
      p.onVarExitScope(for_ref->loop_var);
      return doc;
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<PrimTypeNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) {
      const auto type = Downcast<PrimType>(n);
      return p.WithTirPrefix(runtime::DLDataType2String(type->dtype));
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<TupleTypeNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) {
      const auto type = Downcast<TupleType>(n);
      if (type->fields.empty()) {
        return Doc::Text("None");
      } else {
        std::vector<Doc> fields;
        for (Type field : type->fields) {
          fields.push_back(p.PrintNode(field));
        }
        return p.WithTirPrefix("Tuple") << "[" << Doc::Concat(fields) << "]";
      }
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<BufferNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) {
      Doc doc;
      const Buffer buffer = Downcast<Buffer>(n);
      p.onBufferUsed(buffer);
      doc << buffer->name;
      return doc;
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<BufferStoreNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) {
      Doc doc;
      const BufferStore op = Downcast<BufferStore>(n);
      if (op->indices.size() == 0) {
        doc << p.PrintNode(op->buffer) << "[()] = " << p.PrintNode(op->value);
      } else {
        doc << p.PrintNode(op->buffer) << p.PrintNode(op->indices) << " = "
            << p.PrintNode(op->value);
      }
      return doc;
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<ArrayNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) {
      const auto* node = n.as<ArrayNode>();
      Doc doc;
      doc << '[';
      for (size_t i = 0; i < node->size(); ++i) {
        if (i != 0) {
          doc << ", ";
        }
        doc << p.PrintNode(node->at(i));
      }
      doc << ']';
      return doc;
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<VarNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) {
      Doc doc;
      p.SetExprPrecedence(UnifiedExprPrecedence::kIdentity);
      const Var var = Downcast<Var>(n);
      p.onVarUsed(var);
      doc << var->name_hint;
      return doc;
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<BufferLoadNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) {
      Doc doc;
      p.SetExprPrecedence(UnifiedExprPrecedence::kIdentity);
      const auto buffer_load = Downcast<BufferLoad>(n);
      if (buffer_load->indices.size() == 0) {
        doc << p.PrintNode(buffer_load->buffer) << "[()]";
      } else {
        doc << p.PrintNode(buffer_load->buffer) << p.PrintNode(buffer_load->indices);
      }
      return doc;
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<FloatImmNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) {
      const auto node_ref = Downcast<FloatImm>(n);
      p.SetExprPrecedence(UnifiedExprPrecedence::kIdentity);
      return p.PrintConstScalar<double>(node_ref->dtype, &(node_ref->value));
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<IntImmNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) {
      const auto node_ref = Downcast<IntImm>(n);
      p.SetExprPrecedence(UnifiedExprPrecedence::kIdentity);
      return p.PrintConstScalar<int64_t>(node_ref->dtype, &(node_ref->value));
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<StringObj>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) {
      const auto s = Downcast<String>(n);
      return Doc::StrLiteral(s);
    });

#define TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(OpNode, OpString, OpPrecedence) \
  TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)                            \
      .set_dispatch<OpNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) {    \
        const auto* node = n.as<OpNode>();                                          \
        return p.PrintBinOp(node->a, node->b, OpPrecedence, OpString);              \
      });

TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(MulNode, " * ",
                                            UnifiedExprPrecedence::kMultiplicationDivision)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(DivNode, " / ",
                                            UnifiedExprPrecedence::kMultiplicationDivision)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(FloorDivNode, " // ",
                                            UnifiedExprPrecedence::kMultiplicationDivision)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(FloorModNode, " % ",
                                            UnifiedExprPrecedence::kMultiplicationDivision)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(AddNode, " + ",
                                            UnifiedExprPrecedence::kAdditionSubtraction)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(SubNode, " - ",
                                            UnifiedExprPrecedence::kAdditionSubtraction)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(LTNode, " < ", UnifiedExprPrecedence::kRelational)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(LENode, " <= ", UnifiedExprPrecedence::kRelational)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(GTNode, " > ", UnifiedExprPrecedence::kRelational)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(GENode, " >= ", UnifiedExprPrecedence::kRelational)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(EQNode, " == ", UnifiedExprPrecedence::kEquality)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(NENode, " != ", UnifiedExprPrecedence::kEquality)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(AndNode, " and ", UnifiedExprPrecedence::kAnd)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(OrNode, " or ", UnifiedExprPrecedence::kOr)

String AsTVMScriptUnified(const ObjectRef& node, const String& tir_prefix) {
  auto printer = TVMScriptUnifiedPrinter(tir_prefix);
  Doc content = printer.PrintNode(node);
  Doc doc;
  doc << printer.PrintExtraVarDeclaration() << content;
  return doc.str() + "\n";
}

TVM_REGISTER_GLOBAL("experiment.AsTVMScript").set_body_typed(AsTVMScriptUnified);

}  // namespace tvm
