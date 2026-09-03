# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from lerobot.utils.import_utils import is_package_available, require_package

# Local ACMT-ACT memmap training does not use the Hub Jobs path.  Keep that
# import optional so the lightweight training environment can still invoke
# ``lerobot-train`` without installing the Hub dataset/AV extras.
if is_package_available("datasets"):
    require_package("datasets", extra="dataset")
    from .hf import submit_to_hf
else:
    def submit_to_hf(*args, **kwargs):
        raise ImportError("HF Jobs submission requires the optional 'datasets' extra")

__all__ = ["submit_to_hf"]
