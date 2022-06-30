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

#include "doc_printer.h"

#include "doc.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/expr.h"

namespace tvm {
namespace script {
namespace printer {

namespace {

void SortAndMergeSpans(std::vector<ByteSpan>& spans) {
  if (spans.empty()) {
    return;
  }

  std::sort(spans.begin(), spans.end());
  auto last = spans.begin();
  for (auto cur = spans.begin() + 1; cur != spans.end(); ++cur) {
    if (cur->first > last->second) {
      *++last = *cur;
    } else if (cur->second > last->second) {
      last->second = cur->second;
    }
  }

  spans.erase(++last, spans.end());
}

size_t GetTextWidth(const std::string& text, const ByteSpan& span) {
  // FIXME: this only works for ASCII characters.
  // To do this "correctly", we need to parse UTF-8 into codepoints
  // and call wcwidth() or equivalent for every codepoint.
  size_t ret = 0;
  for (size_t i = span.first; i != span.second; ++i) {
    if (isprint(text[i])) {
      ret += 1;
    }
  }
  return ret;
}

size_t MoveBack(size_t pos, size_t distance) { return distance > pos ? 0 : pos - distance; }

size_t MoveForward(size_t pos, size_t distance, size_t max) {
  return distance > max - pos ? max : pos + distance;
}

size_t GetLineIndex(size_t byte_pos, const std::vector<size_t>& line_starts) {
  auto it = std::upper_bound(line_starts.begin(), line_starts.end(), byte_pos);
  return (it - line_starts.begin()) - 1;
}

using HighlightIter = typename std::vector<ByteSpan>::const_iterator;

ByteSpan PopNextHighlight(HighlightIter& next_highlight, HighlightIter end_highlight) {
  if (next_highlight == end_highlight) {
    return {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
  } else {
    return *next_highlight++;
  }
}

void PrintChunk(const std::pair<size_t, size_t>& lines,
                const std::pair<HighlightIter, HighlightIter>& highlights, const std::string& text,
                const std::vector<size_t>& line_starts, const DocPrinterOptions& options,
                std::string& out) {
  std::ostringstream line_num_printer;

  HighlightIter next_highlight = highlights.first;
  ByteSpan current_highlight = PopNextHighlight(next_highlight, highlights.second);

  for (size_t line_idx = lines.first; line_idx < lines.second; ++line_idx) {
    size_t line_number_width = 0;
    if (options.print_line_numbers) {
      line_num_printer.str("");
      line_num_printer << (line_idx + 1);
      line_num_printer << ' ';
      std::string line_num_str = line_num_printer.str();
      line_number_width = line_num_str.size();
    }

    size_t line_start = line_starts.at(line_idx);
    size_t line_end =
        line_idx + 1 == line_starts.size() ? text.size() : line_starts.at(line_idx + 1);
    out.append(text.begin() + line_start, text.begin() + line_end);

    bool printed_highlight = false;
    size_t line_pos = line_start;
    bool printed_extra_caret = 0;
    while (current_highlight.first < line_end) {
      if (!printed_highlight) {
        out += std::string(line_number_width, ' ');
        printed_highlight = true;
      }

      size_t highlight_end_for_line = std::min(line_end, current_highlight.second);

      size_t num_spaces = GetTextWidth(text, {line_pos, current_highlight.first});
      if (num_spaces > 0 && printed_extra_caret) {
        num_spaces -= 1;
        printed_extra_caret = false;
      }
      out += std::string(num_spaces, ' ');

      size_t num_carets = GetTextWidth(text, {current_highlight.first, highlight_end_for_line});
      if (num_carets == 0 && !printed_extra_caret) {
        // Special case: when highlighting an empty or unprintable string, make sure to print
        // at least one caret still.
        num_carets = 1;
        printed_extra_caret = true;
      } else if (num_carets > 0 && printed_extra_caret) {
        num_carets -= 1;
        printed_extra_caret = false;
      }
      out += std::string(num_carets, '^');

      line_pos = current_highlight.first = highlight_end_for_line;
      if (current_highlight.first == current_highlight.second) {
        current_highlight = PopNextHighlight(next_highlight, highlights.second);
      }
    }

    if (printed_highlight) {
      out.push_back('\n');
    }
  }
}

void PrintCut(size_t num_lines_skipped, std::string& out) {
  if (num_lines_skipped != 0) {
    std::ostringstream s;
    s << "(... " << num_lines_skipped << " lines skipped ...)\n";
    out += s.str();
  }
}

std::pair<size_t, size_t> GetLinesForHighlight(const ByteSpan& highlight,
                                               const std::vector<size_t>& line_starts,
                                               const DocPrinterOptions& options) {
  size_t first_line_of_highlight = GetLineIndex(highlight.first, line_starts);
  size_t first_line_of_chunk = MoveBack(first_line_of_highlight, options.num_context_lines);
  size_t end_line_of_highlight = GetLineIndex(highlight.second - 1, line_starts) + 1;
  size_t end_line_of_chunk =
      MoveForward(end_line_of_highlight, options.num_context_lines, line_starts.size());

  return {first_line_of_chunk, end_line_of_chunk};
}

// If there is only one line between the chunks, it is better to print it as is,
// rather than something like "(... 1 line skipped ...)".
constexpr const size_t kMinLinesToCutOut = 1;

bool TryMergeChunks(std::pair<size_t, size_t>& cur_chunk,
                    const std::pair<size_t, size_t>& new_chunk) {
  if (new_chunk.first <= cur_chunk.second + kMinLinesToCutOut) {
    cur_chunk.second = new_chunk.second;
    return true;
  } else {
    return false;
  }
}

std::string DecorateText(const std::string& text, const std::vector<size_t>& line_starts,
                         const DocPrinterOptions& options,
                         const std::vector<ByteSpan>& highlights) {
  std::string ret;

  if (highlights.empty()) {
    PrintChunk({0, line_starts.size()}, {highlights.begin(), highlights.begin()}, text, line_starts,
               options, ret);
    return ret;
  }

  size_t last_end_line = 0;
  std::pair<size_t, size_t> cur_chunk = GetLinesForHighlight(highlights[0], line_starts, options);
  if (cur_chunk.first <= kMinLinesToCutOut) {
    cur_chunk.first = 0;
  }

  auto first_highlight_in_cur_chunk = highlights.begin();
  for (auto highlight_it = highlights.begin() + 1; highlight_it != highlights.end();
       ++highlight_it) {
    std::pair<size_t, size_t> new_chunk = GetLinesForHighlight(*highlight_it, line_starts, options);

    if (!TryMergeChunks(cur_chunk, new_chunk)) {
      PrintCut(cur_chunk.first - last_end_line, ret);
      PrintChunk(cur_chunk, {first_highlight_in_cur_chunk, highlight_it}, text, line_starts,
                 options, ret);
      last_end_line = cur_chunk.second;
      cur_chunk = new_chunk;
      first_highlight_in_cur_chunk = highlight_it;
    }
  }

  PrintCut(cur_chunk.first - last_end_line, ret);
  if (line_starts.size() - cur_chunk.second <= kMinLinesToCutOut) {
    cur_chunk.second = line_starts.size();
  }
  PrintChunk(cur_chunk, {first_highlight_in_cur_chunk, highlights.end()}, text, line_starts,
             options, ret);
  PrintCut(line_starts.size() - cur_chunk.second, ret);
  return ret;
}

}  // anonymous namespace

DocPrinter::DocPrinter(const DocPrinterOptions& options) : options_(options) {
  line_starts_.push_back(0);
}

void DocPrinter::Clear() {
  output_.str("");
  highlights_.clear();
  line_starts_.resize(1);
}

void DocPrinter::Append(const Doc& doc) { Append(doc, ObjectPath{}); }

void DocPrinter::Append(const Doc& doc, const ObjectPath& path_to_highlight) {
  path_to_highlight_ = path_to_highlight;
  current_path_best_match_length_ = 0;
  current_highlight_candidates_.clear();
  PrintDoc(doc);

  highlights_.insert(highlights_.end(), current_highlight_candidates_.begin(),
                     current_highlight_candidates_.end());
}

String DocPrinter::GetString() const {
  std::string text = output_.str();
  if (!text.empty() && text.back() != '\n') {
    text.push_back('\n');
  }

  std::vector<ByteSpan> highlights = highlights_;
  SortAndMergeSpans(highlights);
  return DecorateText(text, line_starts_, options_, highlights);
}

void DocPrinter::PrintDoc(const Doc& doc) {
  size_t start_pos = output_.tellp();

  if (const auto* doc_node = doc.as<LiteralDocNode>()) {
    PrintTypedDoc(GetRef<LiteralDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<SliceDocNode>()) {
    PrintTypedDoc(GetRef<SliceDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<IdDocNode>()) {
    PrintTypedDoc(GetRef<IdDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<AttrAccessDocNode>()) {
    PrintTypedDoc(GetRef<AttrAccessDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<IndexDocNode>()) {
    PrintTypedDoc(GetRef<IndexDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<OperationDocNode>()) {
    PrintTypedDoc(GetRef<OperationDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<CallDocNode>()) {
    PrintTypedDoc(GetRef<CallDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<LambdaDocNode>()) {
    PrintTypedDoc(GetRef<LambdaDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<ListDocNode>()) {
    PrintTypedDoc(GetRef<ListDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<DictDocNode>()) {
    PrintTypedDoc(GetRef<DictDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<TupleDocNode>()) {
    PrintTypedDoc(GetRef<TupleDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<StmtBlockDocNode>()) {
    PrintTypedDoc(GetRef<StmtBlockDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<ExprStmtDocNode>()) {
    PrintTypedDoc(GetRef<ExprStmtDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<ScopeDocNode>()) {
    PrintTypedDoc(GetRef<ScopeDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<ForDocNode>()) {
    PrintTypedDoc(GetRef<ForDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<AssignDocNode>()) {
    PrintTypedDoc(GetRef<AssignDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<FunctionDocNode>()) {
    PrintTypedDoc(GetRef<FunctionDoc>(doc_node));
  } else {
    LOG(FATAL) << "Do not know how to print " << doc->GetTypeKey();
    throw;
  }

  size_t end_pos = output_.tellp();
  for (const ObjectPath& path : doc->paths) {
    MarkSpan({start_pos, end_pos}, path);
  }
}

void DocPrinter::MarkCurrentPosition(const ObjectPath& path) {
  size_t pos = output_.tellp();
  MarkSpan({pos, pos}, path);
}

void DocPrinter::MarkSpan(const ByteSpan& span, const ObjectPath& path) {
  if (path_to_highlight_.defined()) {
    if (path.Length() >= current_path_best_match_length_ && path.IsPrefixOf(path_to_highlight_)) {
      if (path.Length() > current_path_best_match_length_) {
        current_path_best_match_length_ = path.Length();
        current_highlight_candidates_.clear();
      }
      current_highlight_candidates_.push_back(span);
    }
  }
}

void PythonDocPrinter::PrintTypedDoc(const LiteralDoc& doc) {
  const ObjectRef& value = doc->value;
  if (!value.defined()) {
    output_ << "None";
  } else if (value->IsInstance<FloatImmNode>() || value->IsInstance<IntImmNode>()) {
    PrintNumberNode(Downcast<PrimExpr>(doc->value));
  } else if (const auto* string_obj = value.as<StringObj>()) {
    PrintStringLiteral(GetRef<String>(string_obj));
  } else if (const auto* node = value.as<tir::StringImmNode>()) {
    PrintStringLiteral(node->value);
  } else {
    ICHECK(false) << "Unsupported literal value type " << value->GetTypeKey();
  }
}

void PythonDocPrinter::PrintTypedDoc(const SliceDoc& doc) {
  output_ << "[";
  if (doc->start != nullptr) {
    PrintDoc(doc->start.value());
  }
  output_ << ":";
  if (doc->stop != nullptr) {
    PrintDoc(doc->stop.value());
  }
  output_ << "]";
}

void PythonDocPrinter::PrintTypedDoc(const IdDoc& doc) { output_ << doc->name; }

void PythonDocPrinter::PrintTypedDoc(const AttrAccessDoc& doc) {
  PrintDoc(doc->value);
  output_ << "." << doc->attr;
}

void PythonDocPrinter::PrintTypedDoc(const IndexDoc& doc) {
  PrintDoc(doc->value);
  PrintJoinedElements("[", doc->indices, ", ", "]", "()");
}

void PythonDocPrinter::PrintTypedDoc(const OperationDoc& doc) {
  using OpKind = OperationDocNode::Kind;
  if (doc->kind < OpKind::kUnaryEnd) {
    // TODO: unary op
    throw;
  } else if (doc->kind < OpKind::kBinaryEnd) {
    PrintBinaryOp(doc->kind, doc->operands[0], doc->operands[1]);
  } else {
    PrintSpecialOp(doc->kind, doc->operands);
  }
}

void PythonDocPrinter::PrintTypedDoc(const CallDoc& doc) {
  PrintDoc(doc->callee);

  output_ << "(";

  bool is_first = true;
  for (auto& arg : doc->args) {
    if (is_first) {
      is_first = false;
    } else {
      output_ << ", ";
    }
    PrintDoc(arg);
  }
  for (size_t i = 0; i < doc->kwargs_keys.size(); i++) {
    if (is_first) {
      is_first = false;
    } else {
      output_ << ", ";
    }
    output_ << doc->kwargs_keys[i];
    output_ << "=";
    PrintDoc(doc->kwargs_values[i]);
  }

  output_ << ")";
}

void PythonDocPrinter::PrintTypedDoc(const LambdaDoc& doc) {
  output_ << "lambda ";
  PrintJoinedElements("", doc->args, ", ", ": ");
  PrintDoc(doc->body);
}

void PythonDocPrinter::PrintTypedDoc(const ListDoc& doc) {
  size_t size = doc->elements.size();
  if (size == 0) {
    output_ << "list()";
  } else {
    PrintJoinedElements("[", doc->elements, ", ", "]");
  }
}

void PythonDocPrinter::PrintTypedDoc(const DictDoc& doc) {
  size_t size = doc->keys.size();
  if (size == 0) {
    output_ << "dict()";
  } else {
    output_ << "{";
    size_t idx = 0;
    for (const ExprDoc& key : doc->keys) {
      if (idx > 0) {
        output_ << ", ";
      }
      PrintDoc(key);
      output_ << ": ";
      PrintDoc(doc->values[idx]);
      idx++;
    }
    output_ << "}";
  }
}

void PythonDocPrinter::PrintTypedDoc(const TupleDoc& doc) {
  size_t size = doc->elements.size();
  if (size == 0) {
    output_ << "tuple()";
  } else if (size == 1) {
    output_ << "(";
    PrintDoc(doc->elements[0]);
    output_ << ",)";
  } else {
    PrintJoinedElements("(", doc->elements, ", ", ")");
  }
}

void PythonDocPrinter::PrintTypedDoc(const StmtBlockDoc& doc) { PrintStmtArray(doc->stmts); }

void PythonDocPrinter::PrintTypedDoc(const ExprStmtDoc& doc) { PrintDoc(doc->expr); }

void PythonDocPrinter::PrintTypedDoc(const ScopeDoc& doc) {
  output_ << "with ";
  PrintDoc(doc->rhs);
  if (doc->lhs != nullptr) {
    output_ << " as ";
    PrintDoc(doc->lhs.value());
  }
  output_ << ":";

  PrintStmtBlock(doc->body);
}

void PythonDocPrinter::PrintTypedDoc(const IfDoc& doc) {
  output_ << "if ";
  PrintDoc(doc->predicate);
  output_ << ":";

  if (doc->then_branch.empty()) {
    IncreaseIndent();
    NewLine();
    output_ << "pass";
    DecreaseIndent();
  } else {
    PrintStmtBlock(doc->then_branch);
  }

  if (!doc->else_branch.empty()) {
    output_ << "else:";
    PrintStmtBlock(doc->else_branch);
  }
}

void PythonDocPrinter::PrintTypedDoc(const WhileDoc& doc) {
  output_ << "while ";
  PrintDoc(doc->predicate);
  output_ << ":";

  PrintStmtBlock(doc->body);
}

void PythonDocPrinter::PrintTypedDoc(const ForDoc& doc) {
  output_ << "for ";
  PrintDoc(doc->lhs);
  output_ << " in ";
  PrintDoc(doc->rhs);
  output_ << ":";

  PrintStmtBlock(doc->body);
}

void PythonDocPrinter::PrintTypedDoc(const AssignDoc& doc) {
  // TODO: Assign Kind, Type Annotation
  PrintDoc(doc->lhs);
  if (doc->annotation) {
    output_ << ": ";
    PrintDoc(doc->annotation.value());
  }
  if (doc->rhs) {
    output_ << " = ";
    PrintDoc(doc->rhs.value());
  }
}

void PythonDocPrinter::PrintTypedDoc(const FunctionDoc& doc) {
  for (const ExprDoc& decorator : doc->decorators) {
    output_ << "@";
    PrintDoc(decorator);
    NewLine();
  }
  output_ << "def ";
  PrintDoc(doc->name);
  PrintJoinedElements("(", doc->args, ", ", ")");
  output_ << " -> ";
  PrintDoc(doc->return_type);
  output_ << ":";

  PrintStmtBlock(doc->body);
  NewLine();
}

void PythonDocPrinter::PrintStmtArray(const Array<StmtDoc>& docs) {
  bool first = true;
  for (const StmtDoc& d : docs) {
    if (first) {
      first = false;
    } else {
      NewLine();
    }
    PrintDoc(d);
    for (const ObjectPath& path : d->paths) {
      if (const ArrayIndexPathNode* array_index = path.as<ArrayIndexPathNode>()) {
        MarkCurrentPosition(array_index->GetParent()->MissingArrayElement(array_index->index + 1));
      }
    }
  }
}

void PythonDocPrinter::PrintStmtBlock(const Array<StmtDoc>& docs) {
  IncreaseIndent();
  NewLine();
  PrintStmtArray(docs);
  DecreaseIndent();
}

void PythonDocPrinter::PrintBinaryOp(OperationDocNode::Kind operation_kind, const ExprDoc& a,
                                     const ExprDoc& b) {
  using OpKind = OperationDocNode::Kind;

  PrintDoc(a);
  switch (operation_kind) {
    case OpKind::kAdd:
      output_ << " + ";
      break;
    case OpKind::kSub:
      output_ << " - ";
      break;
    case OpKind::kMul:
      output_ << " * ";
      break;
    case OpKind::kFloorDiv:
      output_ << " // ";
      break;
    case OpKind::kFloorMod:
      output_ << " % ";
      break;
    default:
      LOG(FATAL) << "Unknown Binary Operation " << static_cast<int>(operation_kind);
  }
  PrintDoc(b);
}

void PythonDocPrinter::PrintSpecialOp(OperationDocNode::Kind operation_kind,
                                      const Array<ExprDoc>& operands) {
  using OpKind = OperationDocNode::Kind;

  switch (operation_kind) {
    case OpKind::kAssert:
      output_ << "assert ";
      PrintDoc(operands[0]);
      if (operands.size() > 1) {
        output_ << ", ";
        PrintDoc(operands[1]);
      }
      break;
    default:
      LOG(FATAL) << "Unknown Special Operation " << static_cast<int>(operation_kind);
  }
}

void PythonDocPrinter::PrintStringLiteral(const String& string) {
  // TODO: Escape and smart quote (choose ' or " automatically)
  output_ << "\"" << string << "\"";
}

void PythonDocPrinter::PrintNumberNode(const PrimExpr& expr) {
  const DataType& dtype = expr->dtype;
  std::ostringstream number_value;

  if (const auto* int_node = expr.as<IntImmNode>()) {
    number_value << int_node->value;
  } else if (const auto* float_node = expr.as<FloatImmNode>()) {
    number_value.precision(17);
    number_value << float_node->value;
  } else {
    LOG(FATAL) << "Do not know how to process " << expr->GetTypeKey() << " as literal number";
  }

  if (dtype == DataType::Int(32)) {
    output_ << number_value.str();
  } else if (dtype == DataType::Bool()) {
    output_ << (Downcast<IntImm>(expr)->value ? "True" : "False");
  } else {
    PrintTIRPrimitiveCall(runtime::DLDataType2String(dtype), number_value.str());
  }
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
