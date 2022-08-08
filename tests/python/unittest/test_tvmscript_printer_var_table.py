# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
This file tests the FFI binding of script.printer.VarTable.
These only make sure parameter can be passed to the C++ functions
correctly. The test for the functionality of VarTable is in C++.
"""

import pytest

from tvm.runtime import ObjectPath
from tvm.script.printer.doc import IdDoc
from tvm.script.printer.var_table import VarTable
from tvm.tir import Var


@pytest.mark.parametrize(
    "name_or_factory",
    [
        "a",
        lambda: IdDoc("a"),
    ],
)
def test_define(name_or_factory):
    var_table = VarTable()
    a = Var("a", dtype="int32")
    object_path = ObjectPath.root().attr("a")

    id_doc = var_table.define(a, name_or_factory, object_path)

    assert id_doc.name == "a"
    assert list(id_doc.source_paths) == [object_path]

    id_doc = var_table.get_var_doc(a, object_path)

    assert id_doc.name == "a"
    assert list(id_doc.source_paths) == [object_path]


def test_is_var_defined():
    var_table = VarTable()

    a = Var("a", dtype="int32")
    object_path = ObjectPath.root().attr("a")

    id_doc = var_table.define(a, "a", object_path)

    assert var_table.is_var_defined(a)
    assert a in var_table


def test_remove():
    var_table = VarTable()

    a = Var("a", dtype="int32")
    object_path = ObjectPath.root().attr("a")

    id_doc = var_table.define(a, "a", object_path)
    var_table.remove(a)

    assert not var_table.is_var_defined(a)
    assert a not in var_table
