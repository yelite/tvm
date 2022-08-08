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
"""Functions to print doc into text format"""

from typing import Callable, Optional, Union

from tvm._ffi import register_object
from tvm.runtime import Object, ObjectPath

from . import _ffi_api
from .doc import ExprDoc


@register_object("script.printer.VarTable")
class VarTable(Object):
    """
    Variable Table manages mapping from variable object to ExprDoc during
    the process of printing TVMScript.
    """

    def __init__(self):
        """
        Create an empty VarTable.
        """
        self.__init_handle_by_constructor__(_ffi_api.VarTable)  # type: ignore # pylint: disable=no-member

    def define(
        self,
        obj: Object,
        factory_or_name: Union[str, Callable[[ObjectPath], ExprDoc]],
        object_path: ObjectPath,
    ) -> ExprDoc:
        """
        Define a variable.

        Parameters
        ----------
        obj : Object
            The variable object.
        factory_or_name : Union[str, Callable[[ObjectPath], Doc]]
            The identifier string, or a function that returns doc.
        object_path : ObjectPath
            The object path to be associated with the returned ExprDoc.

        Returns
        -------
        doc : ExprDoc
            The doc for this variable.
        """
        if isinstance(factory_or_name, str):  # pylint: disable=no-else-return
            return _ffi_api.VarTableDefineByName(self, obj, factory_or_name, object_path)
        elif callable(factory_or_name):  # pylint: disable=no-else-return
            return _ffi_api.VarTableDefineByFactory(self, obj, factory_or_name, object_path)
        else:
            raise TypeError(
                f"`factory_or_name` should be str or callable, but it's {type(factory_or_name)}"
            )

    def remove(self, obj: Object):
        """
        Remove a variable.

        Parameters
        ----------
        obj : Object
            The variable object.
        """
        _ffi_api.VarTableRemove(self, obj)

    def get_var_doc(self, obj: Object, object_path: ObjectPath) -> Optional[ExprDoc]:
        """
        Get the doc for a variable.

        Parameters
        ----------
        obj : Object
            The variable object.
        object_path : ObjectPath
            The object path to be associated with the returned ExprDoc.

        Returns
        -------
        doc : ExprDoc
            The doc for this variable.
        """
        return _ffi_api.VarTableGetVarDoc(self, obj, object_path)

    def is_var_defined(self, obj: Object) -> bool:
        """
        Check whether a variable is defined.

        Parameters
        ----------
        obj : Object
            The variable object.

        Returns
        -------
        is_defined : bool
            Whether the variable is defined.
        """
        return _ffi_api.VarTableIsVarDefined(self, obj)

    def __contains__(self, obj: Object) -> bool:
        return self.is_var_defined(obj)
