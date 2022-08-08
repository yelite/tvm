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

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/node/object_path.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/script/printer/traced_object.h>
#include <tvm/script/printer/var_table.h>
#include <tvm/tir/var.h>

#include "tvm/runtime/logging.h"

using namespace tvm;
using namespace tvm::script::printer;

TEST(PrinterVarTableTest, DefineByName) {
  VarTable vars;
  tir::Var x("x");
  ObjectPath object_path = ObjectPath::Root();

  IdDoc doc = vars->Define(x, "x", object_path);

  ICHECK_EQ(doc->name, "x");

  IdDoc second_doc = Downcast<IdDoc>(vars->GetVarDoc(x, object_path).value());

  ICHECK_EQ(second_doc->name, "x");
}

TEST(PrinterVarTableTest, DefineByDocFactory) {
  VarTable vars;
  tir::Var x("x");
  ObjectPath object_path = ObjectPath::Root();

  auto doc_factory = []() { return IdDoc("x"); };

  Doc doc = vars->Define(x, doc_factory, object_path);

  ICHECK_EQ(Downcast<IdDoc>(doc)->name, "x");

  ExprDoc second_doc = vars->GetVarDoc(x, object_path).value();

  ICHECK_EQ(Downcast<IdDoc>(second_doc)->name, "x");
}

TEST(PrinterVarTableTest, RemoveVariable) {
  VarTable vars;
  tir::Var x("x");
  ObjectPath object_path = ObjectPath::Root();

  Doc doc = vars->Define(x, "x", object_path);

  ICHECK(vars->IsVarDefined(x));
  ICHECK(vars->GetVarDoc(x, object_path).defined());

  vars->Remove(x);

  ICHECK(!vars->IsVarDefined(x));
  ICHECK(!vars->GetVarDoc(x, object_path).defined());
}

TEST(PrinterVarTableTest, GetVarDocWithUnknownVariable) {
  VarTable vars;
  tir::Var x("x");
  tir::Var y("y");
  ObjectPath object_path = ObjectPath::Root();

  Doc doc = vars->Define(x, "x", object_path);
  ICHECK(!vars->GetVarDoc(y, object_path).defined());
}

TEST(PrinterVarTableTest, GetVarDocWithObjectPath) {
  VarTable vars;
  tir::Var x("x");
  ObjectPath object_path = ObjectPath::Root();
  ObjectPath second_object_path = ObjectPath::Root()->Attr("x");

  Doc doc = vars->Define(x, "x", object_path);
  ICHECK_EQ(doc->source_paths[0], object_path);
  ICHECK_EQ(doc->source_paths.size(), 1);

  Doc second_doc = vars->GetVarDoc(x, second_object_path).value();
  ICHECK_EQ(second_doc->source_paths[0], second_object_path);
  ICHECK_EQ(second_doc->source_paths.size(), 1);
}

TEST(PrinterVarTableTest, IsVarDefined) {
  VarTable vars;
  tir::Var x("x");
  tir::Var y("y");
  ObjectPath object_path = ObjectPath::Root();

  vars->Define(x, "x", object_path);
  ICHECK(vars->IsVarDefined(x));
  ICHECK(!vars->IsVarDefined(y));
}

TEST(PrinterVarTableTest, DefineDuplicateName_WithString) {
  VarTable vars;
  tir::Var x("x");
  tir::Var y("y");
  ObjectPath object_path = ObjectPath::Root();

  IdDoc x_doc = vars->Define(x, "x", object_path);
  IdDoc y_doc = vars->Define(y, "x", object_path);

  ICHECK_NE(x_doc->name, y_doc->name);
}

TEST(PrinterVarTableTest, DefineDuplicateName_WithFactoryAtFirstDefinition) {
  VarTable vars;
  tir::Var x("x");
  tir::Var y("y");
  ObjectPath object_path = ObjectPath::Root();

  IdDoc y_doc = Downcast<IdDoc>(vars->Define(
      y, []() { return IdDoc("x"); }, object_path));
  IdDoc x_doc = vars->Define(x, "x", object_path);

  ICHECK_NE(x_doc->name, y_doc->name);
}

TEST(PrinterVarTableTest, DefineDuplicateName_WithFactoryAtSecondDefinition) {
  VarTable vars;
  tir::Var x("x");
  tir::Var y("y");
  ObjectPath object_path = ObjectPath::Root();

  IdDoc x_doc = vars->Define(x, "x", object_path);

  // In this case, VarTable can't auto rename the IdDoc to avoid conflict.
  // So an exception is thrown.
  bool failed = false;
  try {
    IdDoc y_doc = Downcast<IdDoc>(vars->Define(
        y, []() { return IdDoc("x"); }, object_path));
  } catch (...) {
    failed = true;
  }
  ASSERT_EQ(failed, true);
}

TEST(PrinterVarTableTest, DefineDuplicateVariable) {
  VarTable vars;
  tir::Var x("x");
  ObjectPath object_path = ObjectPath::Root();

  vars->Define(x, "x", object_path);

  bool failed = false;
  try {
    vars->Define(x, "x", object_path);
  } catch (...) {
    failed = true;
  }
  ASSERT_EQ(failed, true);
}
