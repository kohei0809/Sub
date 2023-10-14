#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.datasets.multi_nav import _try_register_multinavdatasetv1
from habitat.datasets.object_nav import _try_register_objectnavdatasetv1
from habitat.datasets.pointnav import _try_register_pointnavdatasetv1
from habitat.datasets.maximum_info import _try_register_maximuminfodatasetv1, _try_register_maximuminfodatasetv2


def make_dataset(id_dataset, **kwargs):
    logger.info("Initializing dataset {}".format(id_dataset))
    _dataset = registry.get_dataset(id_dataset)
    assert _dataset is not None, "Could not find dataset {}".format(id_dataset)

    return _dataset(**kwargs)


_try_register_multinavdatasetv1()
_try_register_objectnavdatasetv1()
_try_register_pointnavdatasetv1()
_try_register_maximuminfodatasetv1()
_try_register_maximuminfodatasetv2()
