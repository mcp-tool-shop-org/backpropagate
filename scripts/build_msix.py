#!/usr/bin/env python3
"""Build the Microsoft Store MSIX package for backpropagate (1.8.2 track).

Pipeline (each step prints progress; everything under ``--out``):

1. Stage the layout: ``App/python`` (embedded CPython 3.12.10, hash-pinned),
   ``App/python/Lib/site-packages`` (the lock's ``--extra ui`` closure with
   PyPI's CPU-only torch REMOVED, replaced by the pinned cu130 wheel whose
   CUDA runtime is vendored — verified end-to-end with a real CUDA op),
   the UI frontend payload (web.zip + pinned bun, via build_ui_frontend.py),
   ``App/vendor/llama.cpp`` (convert_hf_to_gguf.py + gguf-py, one pinned tag),
   ``App/backprop-launcher.exe`` (csc-built), generated tile/Store logos from
   assets/logo.png, ``THIRD_PARTY_NOTICES.txt``.
2. Gates that refuse to pack: pyproject version must be X.Y.Z (-> X.Y.Z.0,
   4th part is reserved for the Store); torch must report CUDA >= 13.0 and
   run an op on the build GPU; no staged path may exceed MAX_PATH under the
   real WindowsApps install prefix (LongPathsEnabled=0 machines); the staged
   tree must be <= 20 GiB.
3. ``makeappx.exe pack`` -> unsigned .msix (the Store re-signs after
   certification). Self-signing happens ONLY with ``--sideload-test`` (local
   verification), which creates a throwaway cert whose subject matches the
   Partner Center publisher and prints the trust/import instructions.

Requirements on the build machine: Windows, the project venv Python with
Pillow, ``uv`` on PATH, the Windows SDK (makeappx/signtool), ~30 GB scratch.
Network: python.org, download.pytorch.org, github.com (bun + llama.cpp +
licenses), npmjs (the payload's production build).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Pinned inputs. Bump in lockstep with uv.lock / the handoff doc.

PYTHON_VERSION = "3.12.10"
PYTHON_EMBED_URL = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}/"
    f"python-{PYTHON_VERSION}-embed-amd64.zip"
)
PYTHON_EMBED_SHA256 = "4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3"

# PyPI's win_amd64 torch wheel for 2.12.1 is CPU-only (122 MB). The CUDA build
# for Windows lives on download.pytorch.org; cu128 tops out at 2.9.1 there, so
# the minimal >=12.8 variant carrying 2.12.1 is cu130 (CUDA 13.0). The cu130
# win wheel VENDORS the CUDA runtime (no nvidia-* Requires-Dist) — the whole
# GPU stack arrives in this one file.
TORCH_VERSION = "2.12.1"
TORCH_CU_VARIANT = "cu130"
TORCH_WHEEL_URL = (
    "https://download-r2.pytorch.org/whl/cu130/"
    "torch-2.12.1%2Bcu130-cp312-cp312-win_amd64.whl"
)
TORCH_WHEEL_SHA256 = "52c5da6a0898d5d3473c02bd304b7a3bc0b72e351c6f3bfa0783e45ef9f4cd61"
MIN_NVIDIA_DRIVER = 580  # CUDA 13.x runtime requirement; older drivers fall back to CPU silently
_USER_AGENT = "backpropagate-build-msix (+https://github.com/mcp-tool-shop-org/backpropagate)"

LLAMACPP_TAG = "b11323"
LLAMACPP_COMMIT = "f11d642a27b921cf22b6a8beb1b899f960fedcde"
LLAMACPP_ARCHIVE_URL = (
    f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{LLAMACPP_TAG}.zip"
)
# SHA-256 of every file stage_llamacpp copies out of the tag archive:
# LICENSE, convert_hf_to_gguf.py, and every file under conversion/ and
# gguf-py/gguf/ (conversion/ added 2026-10-04, same tag and commit).
# Computed 2026-10-02 from the GitHub tag archive of b11323 after
# https://api.github.com/repos/ggml-org/llama.cpp/git/refs/tags/b11323
# resolved that tag to LLAMACPP_COMMIT. The zip digest is not the pin:
# GitHub does not promise a tag archive stays byte-identical. Regenerate
# with: python scripts/build_msix.py --print-llamacpp-manifest
LLAMACPP_MANIFEST: dict[str, str] = {
    "LICENSE":
        "94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d",
    "conversion/__init__.py":
        "69e73ce6e0cb6c2bf895583b5f60e1af0765bc500799420bbe3e369d3c15bec3",
    "conversion/afmoe.py":
        "d1d91663fd703614b242ddcd6c4f14907068e19539113d5dde7db87258cb46ae",
    "conversion/arctic.py":
        "0f81fda5dfdcca4859c6955270d2d7c352ffd20c0a64c4fc4f557cb24d40b556",
    "conversion/baichuan.py":
        "698f48eab3fffb30226ef8ccd69fd4544ca7cac070fb44e50e70e60d235aae1d",
    "conversion/bailingmoe.py":
        "39c96855c12eef3675640a369e3f6454dddad78486709e2fa3ba60039ff420b9",
    "conversion/bailingmoe3.py":
        "6a29102ec553d62596a74516df27b26392626f6a525f821b684cd596d10773d0",
    "conversion/base.py":
        "b517b5e46a5ee403da8fd2f0fa5ac9b4909f3dd1eadf86043c58a16ba9ede65e",
    "conversion/bert.py":
        "6007377c12d0b05305d380ab6ee3430f97d0a190085758f06d206337002c6638",
    "conversion/bitnet.py":
        "175b64e46105466314e0382ecafb02ebc73db732b8c582604f20ee4f9ab6caa5",
    "conversion/bloom.py":
        "cf17a9c8717c45121c843112dfce7a2fd2aeab060158fb6349f50397f4963c1d",
    "conversion/chameleon.py":
        "96db50d7987eaeb7ac720680577aa1fc97616afc5cb1dbcbf24bd29991a18d34",
    "conversion/chatglm.py":
        "1a01906ca3265438e269137085b2e46c88c6000d56d1d999ef5f62f5b519c01d",
    "conversion/codeshell.py":
        "d717e5059b63090878ec1db6fb2e50cd9cdc6980111e9b3e89e89b00e8b95a00",
    "conversion/cogvlm.py":
        "4d0349fd3276f318a11423a2b6349176dbf3a2dd7bfd76dfe69bae6a26ce22dd",
    "conversion/command_r.py":
        "d1c20f087af2b0d4b761cbc287cf96d599a8696c23934e520b4b798bd3a812a7",
    "conversion/dbrx.py":
        "7b28f1384af62bde5e009c0f90f5f0bf26409af447ad6ce91c8511f8eb5b42d4",
    "conversion/deci.py":
        "1feb845f3430f023f4e9d47777dda5a783eff6c325fba6fb3dce1d1effa1672a",
    "conversion/deepseek.py":
        "09ba15cefd9448d13f27eb0b46462d1a845d7576a6b401f0bf81d73e0419d655",
    "conversion/dots1.py":
        "e730b03e8fdff313907876c8e7a56f405a97009b2bbf0c951e6c0362b9696b54",
    "conversion/dots3.py":
        "e882d5d674f7fb2ac28578e22eee5e7d8b9a9500230ba55271152863d7e195a3",
    "conversion/dotsocr.py":
        "cdd09c97f594b8c411e6df727c71df51df28c8d722c35bd78fdcdd8ffb108d83",
    "conversion/dream.py":
        "5f9b431e76f9f0cb112e9648350d0696807e5ffee386d745f3465384a76ff976",
    "conversion/ernie.py":
        "828db416eef663c28610bf64e6c0d81529558bee16d8fce97d3e54b6157b2519",
    "conversion/exaone.py":
        "9f3dd7859cf59bcf6e900560954fb094e931d603cd3d8b1300ec664ad778b356",
    "conversion/falcon.py":
        "57697368f41af0ddca9278d2fde3c1d200f92efe9296aa61c51bbd7d217b6a42",
    "conversion/falcon_h1.py":
        "950a28acfdf998c0b3c5dca7fc715f342300fd96040f58aa429d5b9d6bbab4d4",
    "conversion/gemma.py":
        "b127a524d92660c86c67a371996ef29076c94cb8bcb9a8c4facf060a901d1671",
    "conversion/glm.py":
        "73a874f5296590af413f865b8cb8ecd54cca0a0c3df7a914a4ee7a7ea47d1b0a",
    "conversion/gpt2.py":
        "b6c57b549c5545fb228b28e863e29a8b449e59d8370f94f0c3c781d40806e03b",
    "conversion/gpt_oss.py":
        "d53ea4177210de1c9ee96d4ace6a2f46ee4640849dbc273e57fc540584faaeb3",
    "conversion/gptneox.py":
        "63e06a691f471828e1cd2671b6dd0749953bb347aab525a260ad4c0b2b81e6c2",
    "conversion/granite.py":
        "917ce34bcdf6089de3e288338485884010e56343db95ad71a526e226958ee23b",
    "conversion/grok.py":
        "7784cb4ad5860e0ddc9667a4379e0dafbf11700553249d87fb5b9819354b1fd1",
    "conversion/grovemoe.py":
        "1a20e35f996016fcaf8dfdaa18ee1e1167001ae13e067ac5c0dd45b7e8525e69",
    "conversion/hrm_text.py":
        "01f6d22c1a626db41d256cac0e43afbab4c80884bf08e8fe2013ba954091da1a",
    "conversion/hunyuan.py":
        "f437578aa60f0b601e960912bdbc664ebe22c814440063b05c15ad4bb1c3681b",
    "conversion/hy_v4.py":
        "0958cc54309901322dd951809cc57a1f9ed33a1aa12cb86653e45ee8e18ca7d0",
    "conversion/internlm.py":
        "ca67d95e16ade174e94ff09bcd0bf6ee7de073d577ee965a69974c2f9ed07ee8",
    "conversion/internvl.py":
        "275186d2ea58e4504fe643190037bb158ea5390e631de43327bcf00f464d95b0",
    "conversion/jais.py":
        "752f826b35c2217019c1ce98dfcd42255cac34a0ad8870d31bdc0aa7da588b39",
    "conversion/jamba.py":
        "1a66ca03190d682a55311c650869299d384227f55cadde7c0d6dc4caa03afbcb",
    "conversion/januspro.py":
        "0ca502908248bacb54a94df515945d2863a9b3174247bcdf046748b45a203280",
    "conversion/kimi_k3.py":
        "72c9767d0a7b9482fca9c1945a3e3d5d3bca440ed8f7e89619b2ced515f0f593",
    "conversion/kimi_linear.py":
        "bc1067ab4deb2326c35a4fcb4e2e8dae7615e65cfdee390d278c5b79f5420b1c",
    "conversion/kimivl.py":
        "73cb871ea06b4dd81ed1dfc62bf27c11091353085fd4f65ed8077ddb72af66c1",
    "conversion/laguna.py":
        "b4535ecd0bfb90f70cd4fa9897ba1116e8d7e8c539c42b75bb9f6037eb5bc392",
    "conversion/lfm2.py":
        "131479370d8d9127d4fe8e7d81453618d80e303488f0343c677596e112761012",
    "conversion/lighton_ocr.py":
        "df7f8030f4469eaa10aac0b70dda8af2a22b2ddfc6af6f41112e03795ed58c77",
    "conversion/llada.py":
        "7e300a3515192e0ff12637b6911d353834e710b69777252f46eee42f1f0829bd",
    "conversion/llama.py":
        "6739a507a74d54a0fb884374da57f6f7844495f233cd1a4d9183cf1a15577eb2",
    "conversion/llama4.py":
        "a39647bb645f2a667effa2998d6be8a71bdbf412f71b8e095b45a7594f20fbbc",
    "conversion/llava.py":
        "4a331b489410d77319443bfd49862f62984440116d3fd7d36b1ac99ebf100906",
    "conversion/maincoder.py":
        "7ce3968578f4dfd62e6b1639d1ab4aa012b9a3f33b78063a7692be79a682e715",
    "conversion/mamba.py":
        "8e11ee0c94754f0471762a6770d953c988bbf3744b6734e669a325460492059d",
    "conversion/maple.py":
        "f82105b61d8479cfd2af2a45107e9340dc78b1ee459ae2bfb50f3b5bfc0eab02",
    "conversion/mellum.py":
        "5852d10e765e67a703ee8b0a75a94061566dc667ebfa325f74ced82ecc1615dc",
    "conversion/mimo.py":
        "f0ab957609eb1807d1d9a4c75d4fff26e2c3f82c7e34224499d99aa247a51b75",
    "conversion/minicpm.py":
        "ef2627437d2649e6615e47ecc92739b0f09ff1ce1e0454d94c639369374d92c9",
    "conversion/minimax.py":
        "dede6725f2da3c3a24fe8146ca6e6f05d4ddf0ebe6b12786e1d0c33593437425",
    "conversion/mistral.py":
        "08b157feb91b3f3d62210252313fc0d45482bc397efd13b9411d40ecce146e21",
    "conversion/mistral3.py":
        "2557b19ac439836ae3deebefe5281b38e8ae4cb204c6f59815288de3e1d250ef",
    "conversion/mpt.py":
        "07513d08befc56233d5520714876bfcf43fdece0f8c8f934249db8dcfe2bae99",
    "conversion/muse_glimmer.py":
        "9bc13418128e4d3404898cb40a5dcffb948ba58e2a5c06c0489f44221b5eda0a",
    "conversion/nanbeige.py":
        "a3256dfb8253eaff6f5f16f36ec03bb072392599829404d1c97ef1df59b9db58",
    "conversion/nemotron.py":
        "d782baa8d4f61cd3e312b0e0d0b0a937fca40a1651212ec5fb47cf9e71ca2d7d",
    "conversion/olmo.py":
        "d97ad239165f720f7c393374cd04d4d1bf7bba55d27ef2111bfa632ccc7cf6b9",
    "conversion/openelm.py":
        "bf93fe8be8611a7d5bdc5ee0a1a724db80f69a2045bbab74debb0e99b02cb73c",
    "conversion/orion.py":
        "1b072a79944a52290b4299b047c0a7f91f7bcb131d332239caa5be86c032802a",
    "conversion/pangu.py":
        "3c69073f05db63a812735d5a94c8ca75f3eb74916f61ed1b3c72c9752bae408e",
    "conversion/phi.py":
        "c8b05d9c91db3a60496b38ffdc2e085e67dec226d15047ab7b1fcf9c0890498c",
    "conversion/pixtral.py":
        "ad39d0eb893e05aaa025d520d92834c6645c186398553ffb396ea5fec079a0fd",
    "conversion/plamo.py":
        "b5dc6817a560950c92eb7a39df0e5a24647155e0b26e70857fa200702bbba8d5",
    "conversion/plm.py":
        "f11eeb8836b675dd26d2f8bb429860821f59fd202a1a36dee1d07179f9b0423e",
    "conversion/pockettts.py":
        "638461f1fde1a147e0fa48194e2a643bd20f801c82718a61795ba84cea8074c7",
    "conversion/qwen.py":
        "084bc3e1daf3a105fb0efadb4d86a43699b0d5dffb56a577c988d76b510e2635",
    "conversion/qwen3tts.py":
        "8919e102f0b8407a39e94e3e93a5033597066d54a48e327d5c5716f066edb3f1",
    "conversion/qwen3vl.py":
        "b6ff12642f41a7f6224a77b4c2d2b47d11b11d807671a0a1c1ab692419e0b67f",
    "conversion/qwen4exp.py":
        "12a0a5aea7877fbb8fe35af041a9c34f8b57b05278871b22c24c650b9760dfc3",
    "conversion/qwenvl.py":
        "ec8fb850a2ebf19191893a861f18f9db4cfc23ff059543036b7227e5feda73ea",
    "conversion/refact.py":
        "af232c6d49ab59e507d5bf672fb34780ff1e1b753402eefb68789d986d94822b",
    "conversion/rwkv.py":
        "0d9c40c1966c5eff5f90c088c42bd7e36ad0a9d789d558b9069d0209743a057e",
    "conversion/sarashina2.py":
        "fd2a42fbe127c4e6c20afa1640f8e9736a8d825e301e8bc12a156cb303a0c0a0",
    "conversion/smallthinker.py":
        "e976485465e39adcfd794764111d5c51bf051cdc7aca888c4790938c2ddb7636",
    "conversion/smolvlm.py":
        "4f069f17c96876ca940fa027fe7a79d05bcb3d18b47694d78d497b3aa8516f53",
    "conversion/spark2_5.py":
        "dfe5873de23884e1686619d6743cfaeca3368ce4687f055f0ad5a23549b89e40",
    "conversion/stablelm.py":
        "d5d382d98299484f3dc5308419bfd67c49cf44e55431f98ea7e68b8392112685",
    "conversion/starcoder.py":
        "d6b83fba5ad11f03d2f904f44ac0e0bac122ef5a392be446bc5dd8ca2ebdc14e",
    "conversion/step3.py":
        "594ae4b97df840ea6790d6cba3710f4454f38675de6c9d064102a23e5e3f62b4",
    "conversion/t5.py":
        "0e28b2ae4d9db47e101a78335bbaf6b067273cd4368da03f92f0c5a805dc780b",
    "conversion/talkie.py":
        "ab06701e8ccba86cae232e8f377d9d877dfe00a84288ec641ae73886241bd48c",
    "conversion/ultravox.py":
        "d23d99493170316f3dae9e1af7a9b5d18a8b7b3c987ccb83280091666c84df38",
    "conversion/wavtokenizer.py":
        "cc6c6fd91cb17070593b37175624c769e904bde584b3e7024906809cfb25a1d8",
    "conversion/xverse.py":
        "c14955d02a33f040d56cbca197de1ac026cf9a74b62f94a376f3c3dd4d866337",
    "conversion/youtuvl.py":
        "e93fb9ebeff14d11efcfeac22bfd408921a67a8b57f12b499d11cd2c368be60e",
    "convert_hf_to_gguf.py":
        "e9a1da876330bbce9687541ab31736542a01b4ac43c6686126514a50f122fb7f",
    "gguf-py/gguf/__init__.py":
        "3ccfc0104cd7ea88c6028743b7bf3f2c89b5f474425de03a217a6072320d7c2f",
    "gguf-py/gguf/constants.py":
        "9fb6729dcc99fae97fe66c9c22fe246455dec36258d816986c32a5d3d74fc47f",
    "gguf-py/gguf/gguf.py":
        "f0c0eeedad0911784b52ffed8e162a0eb5ae6d535ce35705bc16196e46597a72",
    "gguf-py/gguf/gguf_reader.py":
        "d0ea743200e19d7ef0a4c969edcc4a6dba8e3656a790b217e3bdc7b31170718e",
    "gguf-py/gguf/gguf_writer.py":
        "ae45f02b8522e00e054fdc6855cc368068c966a510d05cbcbb6a063620b31b22",
    "gguf-py/gguf/lazy.py":
        "dbc98e3ee9ef8606df34e9d91f98ab29c2697e6824995f1dd8952938c135ad85",
    "gguf-py/gguf/metadata.py":
        "7cedac3b8457271a3f58e5531a7e6958fdecb9ea072f7895ba5d7f6693c9db29",
    "gguf-py/gguf/py.typed":
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "gguf-py/gguf/quants.py":
        "2c927a1b3d9f0920dcf4007fb686e1b0999333e9f65ce43dcc689900c0beae8b",
    "gguf-py/gguf/scripts/gguf_convert_endian.py":
        "6be91de3af4d9b3fc2c90dafb3d1d8eed7e9dd72b3d6b52d245f005c76ece280",
    "gguf-py/gguf/scripts/gguf_dump.py":
        "d8b8fc28e96d15d8a4d6f05cdff4a747f5a06f31172efee9dfc971998ed0203f",
    "gguf-py/gguf/scripts/gguf_editor_gui.py":
        "6e9d70e850268fff5d4108d9ecbc2b4d90e70b61f08e71f7d37ac20916d854bb",
    "gguf-py/gguf/scripts/gguf_hash.py":
        "9f277c9338d12a73857a4e54683d7da4d5de01531751f14401267c9289e4197f",
    "gguf-py/gguf/scripts/gguf_new_metadata.py":
        "972499d35957b4721b9243828d5547d1c84def8d4e6fc641887c6578bf765255",
    "gguf-py/gguf/scripts/gguf_set_metadata.py":
        "c8612a7109428a677ea55976dd5eeed6088934ddef7ef01246a5bc7a6e585b3b",
    "gguf-py/gguf/tensor_mapping.py":
        "04e9d04249cb1e440b08fed170d389604f573f2ae0902c8a96206bfa61389ec7",
    "gguf-py/gguf/utility.py":
        "da920e2c62166ec9f30e407e9fb35ec29fda671310ffbf0f8ce856b62f1c70d7",
    "gguf-py/gguf/vocab.py":
        "353c35e8a52c2afdd113a7a8ce0ddade6cf9b5bb9cf150f965269ea41af3e5fd",
}

# llama-quantize from the same tag's Windows CPU release build. Ollama 0.35
# quantizes no GGUF (measured on 0.35.1, 2026-10-04), so without this binary
# the default q4_k_m export to Ollama ends at f16. The seven files are the
# smallest set it runs with (checked 2026-10-04: its Q4_K_M output was
# byte-identical to the full release folder's); they land next to the
# converter, where export._find_llama_quantize looks first. A release asset
# is an immutable upload, so the zip digest is a pin; each copied file is
# checked as well.
LLAMACPP_BIN_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TAG}/"
    f"llama-{LLAMACPP_TAG}-bin-win-cpu-x64.zip"
)
LLAMACPP_BIN_SHA256 = "bc984e5e0f0337f89c2364cbc24274c31a4ba83f9e8ceb497abb85b8d5046a95"
LLAMACPP_BIN_FILES: dict[str, str] = {
    "llama-quantize.exe":
        "42044e7f4add78082b3f759ebb22190ea99c3f9a9a1b1ed809eb0c988d7f6309",
    "llama-quantize-impl.dll":
        "e6a5d5740d0768384083c3fb01c3be95e9c62b509f3e02e0e24782f7448f06fe",
    "llama.dll":
        "44172200bd26cb7f2cbbd43615b0b7862ed928f882516bcb1874fbb1a0ff6124",
    "llama-common.dll":
        "eca373af97b33b728fbca26c6d2c067608a831fe87626eb59728209c887893b9",
    "ggml.dll":
        "4b55de75c7f98e96f2ddeace2232cdd65d9746833a6556fe276c6d2a02c10e79",
    "ggml-base.dll":
        "0c57672d9e9e8a4e6934d33eda3f2c996e392460e5ad66714d160ae0809d2944",
    "libomp.dll":
        "a12116ba72d1d6820407cf30be23da04ce79d6bb8a71a5ee71759c5a1faa6f1c",
    # libomp's licence (Apache-2.0 WITH LLVM-exception), for the notices.
    "LICENSE-LLVM-OpenMP":
        "fdad1758a9e1f9d5a81e18879b3406772115edc92c24bfa36b70c654f325e8e4",
}

# The GGUF gate's model: a 6 MB random-weight Llama with a sentencepiece
# tokenizer.model, so the converter takes the sentencepiece path. Pinned to a
# commit; every file hash-checked by _fetch.
GGUF_GATE_MODEL_REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
GGUF_GATE_MODEL_REVISION = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
GGUF_GATE_MODEL_FILES: dict[str, str] = {
    "config.json": "5b52782dd9a2b8a4eb87f321448a61cd56cf0764493a1ff73f327259e14ab382",
    "generation_config.json": "571666454de749a224e3739e09821239165ebfed71213632951805ccf25e90bb",
    "model.safetensors": "49c20f32c6c597480fcaec5df2f86c645eabea765cbea1e67886dbae45e5c992",
    "special_tokens_map.json": "4859e5dbde90e059988a0a2136d8df3f2773d4d2fc4c4543690028f0b2166e7f",
    "tokenizer.json": "9cf973f7249a063159ed9c5e7eb05e5d15289b7c5571ae4e1d7315b2189b5831",
    "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
    "tokenizer_config.json": "1847d0a066bd2bd9fb9cb01a2141e7f3befb0a915689878903c3a4dad0f761f9",
}

# Written next to the embedded python.exe. backpropagate.config reads it
# relative to sys.executable; sitecustomize.py (below) also sees it.
STORE_EDITION_MARKER_NAME = "backpropagate-store-edition"
STORE_EDITION_MARKER_TEXT = "store-edition\n"

# Package identity (Partner Center product 9MVXLZVL3TMT). Values are
# case-sensitive; re-read the Publisher on the Product identity page before a
# Store build:
# https://partner.microsoft.com/dashboard/products/9MVXLZVL3TMT/identity
IDENTITY_NAME = "mcp-tool-shop.backpropagate"
IDENTITY_PUBLISHER = "CN=5305D976-6952-4F00-9C21-3A5DB090359F"
PUBLISHER_DISPLAY_NAME = "mcp-tool-shop"
PUBLISHER_ID_SUFFIX = "yn6b8xqrexa5j"
DISPLAY_NAME = "backpropagate"

SIZE_CAP_BYTES = 20 * 1024**3  # 20 GiB; the Store cap is 25 GB (decimal)
MAX_PATH_LIMIT = 259  # usable chars under MAX_PATH=260 (excluding NUL)

# (title, url or None, sha256 of the fetched body or None).
# None url: backpropagate's own LICENSE, or the llama.cpp LICENSE taken
# from the manifest-checked archive. A fetched body is hash-checked by
# _fetch on every build. Digests computed 2026-10-02 from the tag URLs.
NOTICES: list[tuple[str, str | None, str | None]] = [
    (
        "backpropagate (MIT)",
        None,  # local: <repo>/LICENSE
        None,
    ),
    (
        "CPython (PSF License)",
        "https://raw.githubusercontent.com/python/cpython/v3.12.10/LICENSE",
        "3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf",
    ),
    (
        "PyTorch (BSD-3-Clause)",
        "https://raw.githubusercontent.com/pytorch/pytorch/v2.12.1/LICENSE",
        "bd018feef8825e88181c84eb7e3aa4eafb8f08a20d9fd6ef948569610c4a3e43",
    ),
    (
        "bun (MIT)",
        "https://raw.githubusercontent.com/oven-sh/bun/bun-v1.3.13/LICENSE.md",
        "b0e163c004bffb092b08f657a1f6e65b6af64cf62765230717a0ed48a3e562de",
    ),
    (
        "Reflex (MIT)",
        "https://raw.githubusercontent.com/reflex-dev/reflex/v0.9.5.post2/LICENSE",
        "770df32eba7d7f939b0fb92a4b7daf59f85c2267cee763cc991aafb71abf6041",
    ),
    (
        "llama.cpp (MIT)",
        None,  # taken from the extracted archive (shipped in App/vendor too)
        None,
    ),
    (
        "LLVM OpenMP runtime, libomp.dll (Apache-2.0 WITH LLVM-exception)",
        None,  # taken from the llama.cpp release zip (shipped in App/vendor too)
        None,
    ),
]

# Notices with no URL that come from a staged file, by title.
_STAGED_NOTICE_FILES: dict[str, str] = {
    "llama.cpp (MIT)": "App/vendor/llama.cpp/LICENSE",
    "LLVM OpenMP runtime, libomp.dll (Apache-2.0 WITH LLVM-exception)":
        "App/vendor/llama.cpp/LICENSE-LLVM-OpenMP",
}

# --------------------------------------------------------------------------

_LAUNCHER_CS = r"""// backprop-launcher.exe — executes the packaged CPython with the user's args.
// Exists because App Execution Alias extensions cannot bake arguments and a
// packaged python.exe is not directly alias-able; one exe serves both the
// `backprop` and `backpropagate` aliases (identical semantics).
// Build with the OS framework compiler: %WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;

class BackpropLauncher {
    static int Main(string[] args) {
        var dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        var python = Path.Combine(dir, "python", "python.exe");
        if (!File.Exists(python)) {
            Console.Error.WriteLine("backpropagate: packaged python not found at " + python);
            return 1;
        }
        // Ctrl+C is delivered to every process attached to this console. Swallow
        // it here so the launcher survives and keeps waiting on python, whose own
        // Ctrl+C teardown runs; the shell prompt then returns with python's real
        // exit code instead of reappearing mid-shutdown.
        Console.CancelKeyPress += (s, e) => { e.Cancel = true; };
        var sb = new StringBuilder("-m backpropagate");
        foreach (var a in args) { sb.Append(' '); sb.Append(Quote(a)); }
        var psi = new ProcessStartInfo {
            FileName = python,
            Arguments = sb.ToString(),
            UseShellExecute = false,
            WorkingDirectory = Environment.CurrentDirectory,
        };
        try {
            using (var p = Process.Start(psi)) { p.WaitForExit(); return p.ExitCode; }
        } catch (Exception ex) {
            Console.Error.WriteLine("backpropagate: failed to start " + python + ": " + ex.Message);
            return 1;
        }
    }

    // CommandLineToArgvW-compatible quoting: backslashes are literal except
    // in runs that end at a quote or the closing quote (where they double);
    // embedded quotes are backslash-escaped. An empty/whitespace/quote-bearing
    // argument is wrapped in quotes.
    static string Quote(string s) {
        if (s.Length > 0 && s.IndexOfAny(new char[] { ' ', '\t', '"' }) < 0) return s;
        var sb = new StringBuilder("\"");
        int i = 0;
        while (i < s.Length) {
            int slashes = 0;
            while (i < s.Length && s[i] == '\\') { slashes++; i++; }
            if (i == s.Length) { sb.Append('\\', slashes * 2); break; }
            if (s[i] == '"') { sb.Append('\\', slashes * 2 + 1); sb.Append('"'); i++; }
            else { sb.Append('\\', slashes); sb.Append(s[i]); i++; }
        }
        sb.Append('"');
        return sb.ToString();
    }
}
"""

_PTH = "python312.zip\n.\nLib\\site-packages\nimport site\n"

_SITECUSTOMIZE = '''"""backpropagate MSIX site bootstrap (runs at interpreter start via site).

Everything is derived from THIS python's location — no absolute paths are
baked in, because the WindowsApps install prefix changes with every version.
"""

import os as _os
from pathlib import Path as _Path

# .../App/python/Lib/site-packages/sitecustomize.py -> App/python
_python_dir = _Path(__file__).resolve().parents[2]
_app_root = _python_dir.parent

# Store edition: the marker next to python.exe is the source of truth
# (backpropagate.config reads it relative to sys.executable). Overwrite the
# caller's opt-in as well, so the environment this process inherited cannot
# turn model-repository code on. A later assignment in the same process is
# still refused when settings are built, because the marker is in the package.
_marker = _python_dir / "backpropagate-store-edition"
if _marker.is_file():
    _os.environ["BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE"] = "false"

_llama = _app_root / "vendor" / "llama.cpp"
if (_llama / "convert_hf_to_gguf.py").is_file():
    _os.environ.setdefault("BACKPROPAGATE_LLAMA_CPP_PATH", str(_llama))

_payload = _app_root / "ui_frontend_payload"
if (_payload / "payload.json").is_file():
    _os.environ.setdefault("BACKPROPAGATE_UI_PAYLOAD_DIR", str(_payload))
'''

_MANIFEST_TEMPLATE = r"""<?xml version="1.0" encoding="utf-8"?>
<Package
  xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
  xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
  xmlns:uap5="http://schemas.microsoft.com/appx/manifest/uap/windows10/5"
  xmlns:uap10="http://schemas.microsoft.com/appx/manifest/uap/windows10/10"
  xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
  IgnorableNamespaces="uap uap5 uap10 rescap">
  <Identity Name="{identity_name}" Publisher="{publisher}" Version="{msix_version}" ProcessorArchitecture="x64"/>
  <Properties>
    <DisplayName>{display_name}</DisplayName>
    <PublisherDisplayName>{publisher_display}</PublisherDisplayName>
    <Logo>Assets\StoreLogo.png</Logo>
  </Properties>
  <Resources><Resource Language="en-us"/></Resources>
  <Dependencies>
    <!-- uap10:Parameters needs Windows 10 2004 (19041); on older builds it is
         silently ignored and the tile would open a bare python REPL -->
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.19041.0" MaxVersionTested="10.0.26100.0"/>
  </Dependencies>
  <Capabilities>
    <rescap:Capability Name="runFullTrust"/>
  </Capabilities>
  <Applications>
    <Application Id="BackpropagateUI" Executable="App\python\python.exe" EntryPoint="Windows.FullTrustApplication" uap10:Parameters="-m backpropagate ui --open-browser">
      <uap:VisualElements DisplayName="{display_name}" Description="Fine-tune LLMs on your GPU; export to GGUF/Ollama." Square150x150Logo="Assets\Square150x150Logo.png" Square44x44Logo="Assets\Square44x44Logo.png" BackgroundColor="transparent"/>
      <Extensions>
        <uap5:Extension Category="windows.appExecutionAlias" Executable="App\backprop-launcher.exe" EntryPoint="Windows.FullTrustApplication">
          <uap5:AppExecutionAlias>
            <uap5:ExecutionAlias Alias="backprop.exe"/>
            <uap5:ExecutionAlias Alias="backpropagate.exe"/>
          </uap5:AppExecutionAlias>
        </uap5:Extension>
      </Extensions>
    </Application>
  </Applications>
</Package>
"""


# --------------------------------------------------------------------------
# Pure helpers (unit-tested)


def msix_version(pyproject_version: str) -> str:
    """pyproject X.Y.Z -> MSIX X.Y.Z.0 (the 4th part is reserved for the Store)."""
    parts = pyproject_version.split(".")
    if (
        len(parts) != 3
        or not all(p.isdigit() for p in parts)
        or int(parts[0]) == 0
    ):
        raise ValueError(
            f"pyproject version {pyproject_version!r} is not a plain X.Y.Z with "
            "a non-zero major; refusing to map it to an MSIX version"
        )
    return f"{pyproject_version}.0"


def filter_requirements(export_text: str) -> str:
    """Drop the torch stanza from a `uv export` requirements file.

    The lock's torch (PyPI) is CPU-only on win_amd64; the pinned cu130 wheel
    is installed separately. Stanza = a top-level requirement line plus its
    indented hash/comment continuation lines. nvidia-* entries carry
    ``sys_platform == 'linux'`` markers, so they never install on Windows —
    no filtering needed for them.
    """
    kept: list[str] = []
    skipping = False
    for line in export_text.splitlines():
        is_toplevel = line and not line[0].isspace() and not line.startswith("#")
        if is_toplevel:
            skipping = line.startswith("torch==")
        if not skipping:
            kept.append(line)
    return "\n".join(kept) + "\n"


def windowsapps_prefix(msix_ver: str) -> str:
    """The install prefix MSIX deploys to (no trailing separator).

    A fixed Windows target path by construction - built as a plain string
    so the value is identical no matter which OS runs the builder or tests.
    """
    return (
        f"C:\\Program Files\\WindowsApps\\{IDENTITY_NAME}_{msix_ver}_x64__{PUBLISHER_ID_SUFFIX}"
    )


def check_max_path(stage_root: Path, msix_ver: str) -> tuple[int, int]:
    """Return (deepest_package_relative_length, total_under_prefix); raise over budget."""
    deepest_rel, deepest_len = "", 0
    for file in stage_root.rglob("*"):
        if file.is_file():
            rel = len(str(file.relative_to(stage_root)))
            if rel > deepest_len:
                deepest_rel, deepest_len = str(file.relative_to(stage_root)), rel
    total = len(windowsapps_prefix(msix_ver)) + 1 + deepest_len
    if total > MAX_PATH_LIMIT:
        raise RuntimeError(
            f"MAX_PATH gate: {deepest_rel!r} is {deepest_len} chars inside the "
            f"package; under {windowsapps_prefix(msix_ver)!r} that is {total} "
            f"chars > {MAX_PATH_LIMIT} (LongPathsEnabled=0 machines)."
        )
    return deepest_len, total


def check_size(total_bytes: int, cap: int = SIZE_CAP_BYTES) -> None:
    if total_bytes > cap:
        raise RuntimeError(
            f"staged tree is {total_bytes / 1024**3:.1f} GiB > {cap / 1024**3:.0f} GiB cap"
        )


def render_manifest(msix_ver: str) -> str:
    return _MANIFEST_TEMPLATE.format(
        identity_name=IDENTITY_NAME,
        publisher=IDENTITY_PUBLISHER,
        msix_version=msix_ver,
        display_name=DISPLAY_NAME,
        publisher_display=PUBLISHER_DISPLAY_NAME,
    )


def torch_gate_script() -> str:
    """Python source run with the staging interpreter: hard CUDA proof."""
    return (
        "import json, torch\n"
        "cuda = torch.version.cuda or '0'\n"
        "v = tuple(int(p) for p in cuda.split('.')[:2])\n"
        "assert v >= (13, 0), f'torch CUDA variant too old: {cuda}'\n"
        "assert torch.cuda.is_available(), 'CUDA not available on the build GPU'\n"
        "x = torch.randn(64, device='cuda')\n"
        "_ = float((x * x).sum())\n"
        "print(json.dumps({'torch': torch.__version__, 'cuda': cuda, "
        "'device': torch.cuda.get_device_name(0), "
        "'capability': list(torch.cuda.get_device_capability(0))}))\n"
    )


# --------------------------------------------------------------------------
# Build steps


def _log(msg: str) -> None:
    print(f"[build-msix] {msg}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, dest: Path, sha256: str, *, label: str) -> Path:
    """Download with hash-pin; reuse a verified cache hit in dest."""
    if dest.is_file():
        if _sha256_file(dest) == sha256:
            _log(f"{label}: cached {dest.name} (sha256 verified)")
            return dest
        _log(f"{label}: cache hash mismatch -- refetching")
        dest.unlink()
    _log(f"{label}: fetching {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    # download.pytorch.org (Cloudflare) answers 403 to urllib's default
    # User-Agent and serves any other agent string. Seen 2026-10-03.
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=600) as resp:  # noqa: S310 — pinned https URL + hash-checked below  # nosec B310
        with tmp.open("wb") as handle:
            shutil.copyfileobj(resp, handle, 16 * 1024 * 1024)
    actual = _sha256_file(tmp)
    if actual != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{label}: SHA-256 mismatch ({actual} != {sha256}) for {url}")
    os.replace(tmp, dest)
    _log(f"{label}: {dest.name} ({dest.stat().st_size / 1024**2:.0f} MiB, hash OK)")
    return dest


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    _log("run: " + " ".join(cmd)[:200])
    return subprocess.run(cmd, check=True, **kwargs)  # nosec B603 — fixed internal argv


def _project_version(repo: Path) -> str:
    import tomllib

    return tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]


def _find_sdk_tool(name: str) -> Path:
    kits = Path("C:/Program Files (x86)/Windows Kits/10/bin")
    if not kits.is_dir():
        raise RuntimeError(f"Windows SDK not found under {kits}")
    best: tuple[tuple[int, ...], Path] | None = None
    for candidate in kits.glob("*/x64/" + name):
        try:
            ver = tuple(int(p) for p in candidate.parent.parent.name.split("."))
        except ValueError:
            continue
        if best is None or ver > best[0]:
            best = (ver, candidate)
    if best is None:
        raise RuntimeError(f"{name} not found under {kits}/*/x64")
    return best[1]


def _find_csc() -> Path:
    for root in (
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Microsoft.NET/Framework64",
    ):
        candidates = sorted(root.glob("v*/csc.exe"), reverse=True)
        if candidates:
            return candidates[0]
    raise RuntimeError("no framework64 csc.exe found")


def stage_python(downloads: Path, stage: Path) -> Path:
    """Embedded CPython with a working ._pth; returns python.exe path."""
    zip_path = _fetch(PYTHON_EMBED_URL, downloads / f"python-{PYTHON_VERSION}-embed-amd64.zip", PYTHON_EMBED_SHA256, label="cpython-embed")
    py_home = stage / "App" / "python"
    if py_home.exists():
        shutil.rmtree(py_home)
    py_home.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(py_home)
    (py_home / "python312._pth").write_text(_PTH, encoding="utf-8")
    exe = py_home / "python.exe"
    result = _run([str(exe), "-c", "import sys, site; print(sys.version)"], capture_output=True, text=True)
    _log(f"embedded python: {result.stdout.strip().splitlines()[0]}")
    stage_store_edition_marker(py_home)
    return exe


def stage_store_edition_marker(python_home: Path) -> Path:
    """Write the Store-edition marker next to the embedded interpreter.

    ``backpropagate.config.is_store_edition`` reads this file relative to
    ``sys.executable``. It lives inside the package, so clearing an
    environment variable cannot turn the edition off.
    """
    dest = Path(python_home) / STORE_EDITION_MARKER_NAME
    dest.write_text(STORE_EDITION_MARKER_TEXT, encoding="utf-8")
    _log(f"store edition: marker {dest.name}")
    return dest


def install_dependencies(repo: Path, downloads: Path, stage: Path, python_exe: Path) -> dict:
    """Lock-closure minus PyPI torch, plus the pinned cu130 wheel, plus our own wheel."""
    export_path = downloads / "requirements-export.txt"
    _run(
        ["uv", "export", "--frozen", "--no-emit-project", "--extra", "ui",
         "--format", "requirements-txt", "-o", str(export_path)],
        cwd=str(repo),
    )
    filtered = filter_requirements(export_path.read_text(encoding="utf-8"))
    filtered_path = downloads / "requirements-filtered.txt"
    filtered_path.write_text(filtered, encoding="utf-8")
    kept_torch = [ln for ln in filtered.splitlines() if ln.startswith("torch==")]
    if kept_torch:
        raise RuntimeError("torch stanza survived the export filter — refusing")
    _run(
        ["uv", "pip", "install", "--python", str(python_exe), "--no-deps",
         "--require-hashes", "-r", str(filtered_path)],
        cwd=str(repo),
    )
    wheel = _fetch(TORCH_WHEEL_URL, downloads / "torch-2.12.1+cu130-cp312-cp312-win_amd64.whl", TORCH_WHEEL_SHA256, label="torch-cu130")
    _run(
        ["uv", "pip", "install", "--python", str(python_exe), "--no-deps", str(wheel)],
    )
    wheel_out = downloads / "dist-out"
    if wheel_out.exists():
        shutil.rmtree(wheel_out)
    _run(["uv", "build", "--wheel", "--out-dir", str(wheel_out)], cwd=str(repo))
    wheels = list(wheel_out.glob("backpropagate-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one built wheel, got {wheels}")
    _run(
        ["uv", "pip", "install", "--python", str(python_exe), "--no-deps", str(wheels[0])],
    )
    _run(["uv", "pip", "check", "--python", str(python_exe)])
    site_packages = stage / "App" / "python" / "Lib" / "site-packages"
    (site_packages / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    return {"built_wheel": wheels[0].name}


def gate_torch(python_exe: Path) -> dict:
    result = _run(
        [str(python_exe), "-c", torch_gate_script()], capture_output=True, text=True
    )
    info = json.loads(result.stdout.strip().splitlines()[-1])
    _log(f"torch gate: {info}")
    return info


def stage_payload(repo: Path, stage: Path, python_exe: Path, work: Path, reuse: Path | None) -> dict:
    if reuse is not None:
        _log(f"payload: reusing {reuse}")
        shutil.copytree(reuse, work / "ui_frontend_payload")
    else:
        result = _run(
            [str(python_exe), str(repo / "scripts" / "build_ui_frontend.py"), "--out", str(work)],
            cwd=str(repo),
        )
        del result
    payload = work / "ui_frontend_payload"
    meta = json.loads((payload / "payload.json").read_text(encoding="utf-8"))
    target = stage / "App" / "ui_frontend_payload"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(payload, target)
    return {"payload": meta}


def _llamacpp_root(tag: str) -> str:
    return f"llama.cpp-{tag}/"


def _copied_llamacpp_rel(rel: str) -> bool:
    # The converter is a thin entry point since the 2026 split: the model
    # classes live in conversion/*.py next to it (`from conversion import ...`).
    # 1.8.2 copied only the script, so the packaged converter could not start.
    return (
        rel == "LICENSE"
        or rel == "convert_hf_to_gguf.py"
        or (rel.startswith("conversion/") and not rel.endswith("/"))
        or (rel.startswith("gguf-py/gguf/") and not rel.endswith("/"))
    )


def llamacpp_archive_members(zf: zipfile.ZipFile, tag: str) -> dict[str, bytes]:
    """Files this build copies: LICENSE, the converter, conversion/*, gguf-py/gguf/*."""
    root = _llamacpp_root(tag)
    files: dict[str, bytes] = {}
    for name in zf.namelist():
        if not name.startswith(root) or name.endswith("/"):
            continue
        rel = name[len(root):]
        if not _copied_llamacpp_rel(rel):
            continue
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise RuntimeError(f"llama.cpp archive has an unsafe path: {rel}")
        files[rel] = zf.read(name)
    return files


def verify_llamacpp_manifest(files: dict[str, bytes], manifest: dict[str, str]) -> None:
    """Fail if the copied set is not exactly the pinned path/sha256 map."""
    extra = sorted(set(files) - set(manifest))
    missing = sorted(set(manifest) - set(files))
    changed = sorted(
        rel
        for rel, data in files.items()
        if rel in manifest and hashlib.sha256(data).hexdigest() != manifest[rel]
    )
    if not (extra or missing or changed):
        return
    parts: list[str] = []
    if changed:
        parts.append("changed: " + ", ".join(changed))
    if extra:
        parts.append("not in manifest: " + ", ".join(extra))
    if missing:
        parts.append("missing: " + ", ".join(missing))
    raise RuntimeError("llama.cpp manifest mismatch (" + "; ".join(parts) + ")")


def format_llamacpp_manifest(
    files: dict[str, bytes], *, tag: str | None = None, commit: str | None = None
) -> str:
    """Sorted ``path sha256`` lines, optional tag/commit comment first."""
    lines: list[str] = []
    if tag is not None:
        lines.append(f"# tag {tag} commit {commit or 'unverified'}")
    for rel, data in sorted(files.items()):
        lines.append(f"{rel} {hashlib.sha256(data).hexdigest()}")
    return "\n".join(lines) + ("\n" if lines else "")


def llamacpp_manifest_text(archive: Path, tag: str, *, commit: str | None = None) -> str:
    with zipfile.ZipFile(archive) as zf:
        files = llamacpp_archive_members(zf, tag)
    return format_llamacpp_manifest(files, tag=tag, commit=commit)


def verify_llamacpp_tag_commit(tag: str, commit: str) -> None:
    """The tag ref must still point at the recorded commit. Run on fetch."""
    url = f"https://api.github.com/repos/ggml-org/llama.cpp/git/refs/tags/{tag}"
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 — fixed https URL  # nosec B310
        sha = json.loads(resp.read().decode())["object"]["sha"]
    if sha != commit:
        raise RuntimeError(f"llama.cpp tag {tag} moved: {sha} != pinned {commit}")


def fetch_llamacpp_archive(dest: Path, tag: str, *, commit: str | None) -> Path:
    """Download the tag zip. When ``commit`` is set, check the tag ref first."""
    if commit is not None:
        verify_llamacpp_tag_commit(tag, commit)
        _log(f"llama.cpp: fetching archive {tag} (commit verified)")
    else:
        _log(f"llama.cpp: fetching archive {tag}")
    if tag == LLAMACPP_TAG:
        url = LLAMACPP_ARCHIVE_URL
    else:
        url = f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{tag}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    # The tag ref was checked when a commit pin was supplied. Every copied
    # file is hash-checked by verify_llamacpp_manifest after this returns.
    with urllib.request.urlopen(url, timeout=300) as resp:  # noqa: S310  # nosec B310 — tag ref checked when pinned; copied-file hashes checked by caller
        data = resp.read()
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return dest


def stage_llamacpp_files(files: dict[str, bytes], stage: Path) -> None:
    vendor = stage / "App" / "vendor" / "llama.cpp"
    if vendor.exists():
        shutil.rmtree(vendor)
    for rel, data in sorted(files.items()):
        dest = vendor / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)


def stage_llamacpp(downloads: Path, stage: Path) -> dict:
    archive = downloads / f"llama.cpp-{LLAMACPP_TAG}.zip"
    if not archive.is_file():
        fetch_llamacpp_archive(archive, LLAMACPP_TAG, commit=LLAMACPP_COMMIT)
    with zipfile.ZipFile(archive) as zf:
        files = llamacpp_archive_members(zf, LLAMACPP_TAG)
    # Cached archive or a fresh download: every copied byte is checked.
    verify_llamacpp_manifest(files, LLAMACPP_MANIFEST)
    stage_llamacpp_files(files, stage)
    _log(f"llama.cpp: {LLAMACPP_TAG} staged ({len(files)} files, manifest verified)")
    return {"llama_cpp": {"tag": LLAMACPP_TAG, "commit": LLAMACPP_COMMIT}}


def llamacpp_bin_members(zf: zipfile.ZipFile) -> dict[str, bytes]:
    """The LLAMACPP_BIN_FILES entries from the release zip (they sit at its root)."""
    present = set(zf.namelist())
    files: dict[str, bytes] = {}
    for name in LLAMACPP_BIN_FILES:
        if name not in present:
            raise RuntimeError(f"llama.cpp release zip has no {name}")
        files[name] = zf.read(name)
    return files


def stage_llamacpp_quantize(downloads: Path, stage: Path) -> dict:
    """llama-quantize and its DLLs, next to the staged converter. Run after stage_llamacpp."""
    archive = _fetch(
        LLAMACPP_BIN_URL,
        downloads / f"llama-{LLAMACPP_TAG}-bin-win-cpu-x64.zip",
        LLAMACPP_BIN_SHA256,
        label="llama.cpp-bin",
    )
    with zipfile.ZipFile(archive) as zf:
        files = llamacpp_bin_members(zf)
    verify_llamacpp_manifest(files, LLAMACPP_BIN_FILES)
    vendor = stage / "App" / "vendor" / "llama.cpp"
    for name, data in sorted(files.items()):
        (vendor / name).write_bytes(data)
    _log(f"llama.cpp: llama-quantize staged ({len(files)} files, hashes verified)")
    return {"llama_quantize_files": len(files)}


def gguf_gate_script() -> str:
    """Python source run with the staged interpreter: a real q4_k_m GGUF export.

    It goes through backpropagate's own export_gguf, so it needs what a user's
    export needs: sitecustomize pointing at the vendored converter, the
    converter's conversion/ package, sentencepiece, and llama-quantize.
    """
    return (
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "src, out = Path(sys.argv[1]), Path(sys.argv[2])\n"
        "llama = os.environ.get('BACKPROPAGATE_LLAMA_CPP_PATH', '')\n"
        "assert llama, 'sitecustomize did not set BACKPROPAGATE_LLAMA_CPP_PATH'\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "from backpropagate.export import export_gguf\n"
        "model = AutoModelForCausalLM.from_pretrained(src)\n"
        "tok = AutoTokenizer.from_pretrained(src)\n"
        "# The fixture's config carries pad_token_id -1, which the converter rejects.\n"
        "model.config.pad_token_id = None\n"
        "model.generation_config.pad_token_id = None\n"
        "r = export_gguf(model=model, tokenizer=tok, output_dir=out, quantization='q4_k_m',\n"
        "                model_name='gate', emit_model_card=False)\n"
        "p = Path(r.path)\n"
        "assert p.read_bytes()[:4] == b'GGUF', f'not a GGUF file: {p}'\n"
        "assert r.quantization == 'q4_k_m' and not r.deferred_quantization, (\n"
        "    'llama-quantize was not used', r.quantization, r.deferred_quantization)\n"
        "print(json.dumps({'llama_cpp': llama, 'gguf': p.name, 'bytes': p.stat().st_size,\n"
        "                  'quantization': r.quantization}))\n"
    )


def gate_gguf_export(python_exe: Path, stage: Path, downloads: Path, work: Path) -> dict:
    """Export the pinned tiny model to q4_k_m GGUF with the staged Python.

    1.8.2 passed every gate and still could not export a GGUF: the converter's
    conversion/ package, sentencepiece and a working quantizer were all
    missing. This gate runs the export a user runs, so any of those fails here.
    """
    model_dir = downloads / "gguf-gate-model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in GGUF_GATE_MODEL_FILES.items():
        url = (
            f"https://huggingface.co/{GGUF_GATE_MODEL_REPO}/resolve/"
            f"{GGUF_GATE_MODEL_REVISION}/{name}"
        )
        _fetch(url, model_dir / name, digest, label=f"gguf-gate:{name}")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    # The staged sitecustomize must supply the llama.cpp path, so drop the
    # builder's own BACKPROPAGATE_* settings; keep the stage free of .pyc files.
    env = {k: v for k, v in os.environ.items() if not k.startswith("BACKPROPAGATE_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    cmd = [str(python_exe), "-c", gguf_gate_script(), str(model_dir), str(work / "out")]
    _log("run: gguf gate (q4_k_m export of the tiny model with the staged python)")
    result = subprocess.run(  # nosec B603 - fixed internal argv
        cmd, cwd=str(work), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        tail = (result.stdout + result.stderr)[-3000:]
        raise RuntimeError(f"GGUF gate failed (exit {result.returncode}):\n{tail}")
    info = json.loads(result.stdout.strip().splitlines()[-1])
    vendor = stage / "App" / "vendor" / "llama.cpp"
    if Path(info["llama_cpp"]).resolve() != vendor.resolve():
        raise RuntimeError(f"GGUF gate used {info['llama_cpp']}, not the staged {vendor}")
    _log(f"gguf gate: {info}")
    return info


def render_llamacpp_manifest_for_tag(tag: str) -> str:
    """Fetch ``tag`` and return the manifest text for the next pin bump.

    The pinned tag is commit-checked. Another tag is fetched as-is; the
    comment line carries that tag's current commit when the ref API answers.
    """
    commit = LLAMACPP_COMMIT if tag == LLAMACPP_TAG else None
    dest = Path(tempfile.gettempdir()) / f"llama.cpp-{tag}-manifest.zip"
    if dest.exists():
        dest.unlink()
    fetch_llamacpp_archive(dest, tag, commit=commit)
    resolved = commit
    if resolved is None:
        try:
            url = f"https://api.github.com/repos/ggml-org/llama.cpp/git/refs/tags/{tag}"
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 — fixed https API  # nosec B310
                resolved = json.loads(resp.read().decode())["object"]["sha"]
        except (OSError, ValueError, KeyError):
            resolved = None
    return llamacpp_manifest_text(dest, tag, commit=resolved)


def stage_launcher(stage: Path, downloads: Path) -> None:
    csc = _find_csc()
    src = downloads / "backprop_launcher.cs"
    src.write_text(_LAUNCHER_CS, encoding="utf-8")
    out = stage / "App" / "backprop-launcher.exe"
    _run([str(csc), "/nologo", "/target:exe", f"/out:{out}", str(src)])


def stage_assets(repo: Path, stage: Path) -> None:
    from PIL import Image

    src = Image.open(repo / "assets" / "logo.png").convert("RGBA")
    bg = src.getpixel((0, 0))
    if bg[3] == 0:
        bg = (0x0B, 0x0E, 0x14, 255)  # dark fallback when the logo is transparent-edged
    canvas_w = max(src.size)
    canvas = Image.new("RGBA", (canvas_w, canvas_w), bg)
    canvas.paste(src, ((canvas_w - src.width) // 2, (canvas_w - src.height) // 2), src)
    assets = stage / "Assets"
    assets.mkdir(exist_ok=True)
    targets = [
        ("StoreLogo.png", (50, 50)),
        ("Square44x44Logo.png", (44, 44)),
        ("Square71x71Logo.png", (71, 71)),
        ("Square150x150Logo.png", (150, 150)),
        ("Square310x310Logo.png", (310, 310)),
        ("Wide310x150Logo.png", (310, 150)),
        ("Square44x44Logo.targetsize-256.png", (256, 256)),
        ("Square44x44Logo.targetsize-48.png", (48, 48)),
        ("Square44x44Logo.targetsize-32.png", (32, 32)),
        ("Square44x44Logo.targetsize-24.png", (24, 24)),
        ("Square44x44Logo.targetsize-16.png", (16, 16)),
    ]
    for name, size in targets:
        resized = canvas.resize(size, Image.LANCZOS)
        resized.convert("RGB").save(assets / name) if name != "StoreLogo.png" else resized.save(assets / name)
    _log(f"assets: {len(targets)} logos generated (pad color {bg})")


def _notice_cache_name(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower()
    return f"notice-{slug}.txt"


def stage_notices(repo: Path, stage: Path, downloads: Path) -> None:
    parts: list[str] = []
    downloads.mkdir(parents=True, exist_ok=True)
    for title, source, digest in NOTICES:
        if source is None and title.startswith("backpropagate"):
            text = (repo / "LICENSE").read_text(encoding="utf-8")
        elif source is None:  # llama.cpp and libomp — from the staged vendor copy
            text = (stage / _STAGED_NOTICE_FILES[title]).read_text(encoding="utf-8")
        else:
            if not digest:
                raise RuntimeError(f"notice {title!r} is fetched but has no SHA-256 pin")
            dest = downloads / _notice_cache_name(title)
            _fetch(source, dest, digest, label=f"notice:{title}")
            text = dest.read_text(encoding="utf-8")
        parts.append(f"{'=' * 78}\n{title}\n{'=' * 78}\n\n{text.strip()}\n")
    (stage / "THIRD_PARTY_NOTICES.txt").write_text(
        "backpropagate Microsoft Store package — third-party notices\n\n"
        + "\n".join(parts),
        encoding="utf-8",
    )
    _log(f"notices: {len(NOTICES)} license texts")


def write_manifest_and_info(stage: Path, msix_ver: str, meta: dict, total_bytes: int, deepest: int, deepest_total: int) -> dict:
    (stage / "AppxManifest.xml").write_text(render_manifest(msix_ver), encoding="utf-8")
    info = {
        "schema": 1,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "identity": {
            "name": IDENTITY_NAME,
            "publisher": IDENTITY_PUBLISHER,
            "publisher_display": PUBLISHER_DISPLAY_NAME,
            "package_family_suffix": PUBLISHER_ID_SUFFIX,
        },
        "version": meta["version"],
        "msix_version": msix_ver,
        "python": {
            "version": PYTHON_VERSION,
            "url": PYTHON_EMBED_URL,
            "sha256": PYTHON_EMBED_SHA256,
        },
        "torch": {
            "version": f"{TORCH_VERSION}+{TORCH_CU_VARIANT}",
            "url": TORCH_WHEEL_URL,
            "sha256": TORCH_WHEEL_SHA256,
            "min_nvidia_driver": MIN_NVIDIA_DRIVER,
            "gate": meta.get("torch_gate"),
        },
        "llama_cpp": {
            "tag": LLAMACPP_TAG,
            "commit": LLAMACPP_COMMIT,
            "files": len(LLAMACPP_MANIFEST),
            "quantize_zip_sha256": LLAMACPP_BIN_SHA256,
            "quantize_files": len(LLAMACPP_BIN_FILES),
            "gguf_gate": meta.get("gguf_gate"),
        },
        "ui_payload": meta.get("payload"),
        "built_wheel": meta.get("built_wheel"),
        "size_bytes": total_bytes,
        "maxpath": {"deepest_relative": deepest, "total_under_install_prefix": deepest_total},
    }
    (stage / "App" / "build-info.json").write_text(
        json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return info


def pack(stage: Path, out_dir: Path, unsigned_name: str) -> Path:
    makeappx = _find_sdk_tool("makeappx.exe")
    dest = out_dir / unsigned_name
    if dest.exists():
        dest.unlink()
    _run([str(makeappx), "pack", "/d", str(stage), "/p", str(dest), "/o"])
    _log(f"packed: {dest} ({dest.stat().st_size / 1024**3:.2f} GiB)")
    return dest


def sideload_sign(msix: Path) -> None:
    """Self-sign for LOCAL verification only. The cert subject matches the
    Partner Center publisher so the signature validates against the manifest.
    The signing key is destroyed immediately after signing (the .pfx is
    deleted and the cert is removed from Cert:\\CurrentUser\\My with
    -DeleteKey); only the public .cer remains, for the trust-store import
    printed below. The trust/import commands are for a human admin to run -
    this script never touches certificate stores beyond its own throwaway cert.
    """
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if pwsh is None:
        raise RuntimeError("no pwsh/powershell found for self-signing")
    # Find signtool before any key exists, so a missing SDK cannot strand one.
    signtool = _find_sdk_tool("signtool.exe")
    cert_dir = msix.parent / "sideload-cert"
    cert_dir.mkdir(exist_ok=True)
    pfx = cert_dir / "backpropagate-sideload.pfx"
    cer = cert_dir / "backpropagate-sideload.cer"
    password = "sideload-test"  # nosec B105 - throwaway password for the local self-signed test cert, not a shipped secret
    friendly = "backpropagate sideload test"
    create = (
        "$p = ConvertTo-SecureString -String '" + password + "' -Force -AsPlainText; "
        f"$c = New-SelfSignedCertificate -Type Custom -Subject '{IDENTITY_PUBLISHER}' "
        f"-KeyUsage DigitalSignature -FriendlyName '{friendly}' "
        "-CertStoreLocation Cert:\\CurrentUser\\My "
        "-TextExtension @('2.5.29.37={text}1.3.6.1.5.5.7.3.3'); "
        f"Export-PfxCertificate -Cert $c -FilePath '{pfx}' -Password $p | Out-Null; "
        f"Export-Certificate -Cert $c -FilePath '{cer}' | Out-Null; "
        "Write-Output $c.Thumbprint"
    )
    thumbprint: str | None = None
    try:
        created = subprocess.run(  # nosec B603 - fixed internal argv
            [pwsh, "-NoProfile", "-Command", create], check=True, capture_output=True, text=True
        )
        thumbprints = re.findall(r"\b[0-9A-Fa-f]{40}\b", created.stdout)
        if not thumbprints:
            raise RuntimeError(
                "could not read the self-signed cert thumbprint from pwsh output: "
                f"{created.stdout!r}"
            )
        thumbprint = thumbprints[-1].upper()
        _run([str(signtool), "sign", "/fd", "sha256", "/a", "/f", str(pfx), "/p", password, str(msix)])
    finally:
        # Destroy the signing key on every exit path (signing failed, the
        # thumbprint was unreadable, the export half-ran): a usable key for the
        # Store publisher CN must not remain on disk. -DeleteKey wipes the key
        # material along with the cert. Without a thumbprint, remove this
        # script's own throwaway certs by their friendly name instead.
        pfx.unlink(missing_ok=True)
        if thumbprint is not None:
            remove = f"Remove-Item 'Cert:\\CurrentUser\\My\\{thumbprint}' -DeleteKey -Force"
        else:
            remove = (
                "Get-ChildItem Cert:\\CurrentUser\\My | "
                f"Where-Object {{ $_.FriendlyName -eq '{friendly}' }} | "
                "Remove-Item -DeleteKey -Force"
            )
        subprocess.run(  # nosec B603 - fixed internal argv; removes only certs this script created
            [pwsh, "-NoProfile", "-Command", remove],
            check=True, capture_output=True, text=True,
        )
    print(
        f"\nSIGNED FOR SIDELOAD TESTING ONLY (throwaway cert {thumbprint}; its private\n"
        "key was deleted right after signing and is unrecoverable).\n"
        "Have an admin run:\n"
        f"  certutil -addstore TrustedPeople \"{cer}\"\n"
        f"  Add-AppxPackage \"{msix}\"\n"
        "After testing, remove every trace:\n"
        "  Remove-AppxPackage -Package (Get-AppxPackage *backpropagate*).PackageFullName\n"
        f"  certutil -delstore TrustedPeople {thumbprint}\n"
        f"  Remove-Item \"{cer}\"\n"
        "The Store build is the UNSIGNED .msix -- Partner Center re-signs it.\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--out", type=Path, default=None, help="output dir for the msix + caches")
    parser.add_argument("--reuse-payload", type=Path, default=None,
                        help="reuse an existing ui_frontend_payload dir instead of rebuilding")
    parser.add_argument("--sideload-test", action="store_true",
                        help="self-sign the msix for LOCAL sideload verification (never for the Store upload)")
    parser.add_argument("--pack-only", type=Path, default=None,
                        help="skip staging; pack an existing stage dir (gates re-run)")
    parser.add_argument(
        "--print-llamacpp-manifest",
        action="store_true",
        help="fetch a llama.cpp tag and print the copied-file SHA-256 manifest, then exit",
    )
    parser.add_argument(
        "--llamacpp-tag",
        default=None,
        help="tag for --print-llamacpp-manifest (default: the pinned tag)",
    )
    args = parser.parse_args(argv)

    if args.print_llamacpp_manifest:
        tag = args.llamacpp_tag or LLAMACPP_TAG
        sys.stdout.write(render_llamacpp_manifest_for_tag(tag))
        return 0

    if os.name != "nt":
        print("build_msix.py must run on Windows", file=sys.stderr)
        return 2
    if args.out is None:
        print("build_msix.py: --out is required", file=sys.stderr)
        return 2
    t0 = time.monotonic()
    repo = args.repo.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    downloads = out_dir / "_downloads"
    downloads.mkdir(exist_ok=True)

    version = _project_version(repo)
    msix_ver = msix_version(version)
    _log(f"version {version} -> msix {msix_ver}")

    stage = out_dir / "stage"
    meta: dict = {"version": version}

    if args.pack_only is not None:
        stage = args.pack_only.resolve()
        if not stage.is_dir():
            print(f"--pack-only stage dir missing: {stage}", file=sys.stderr)
            return 2
        _log(f"pack-only: reusing stage {stage}")
    else:
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        python_exe = stage_python(downloads, stage)
        meta.update(install_dependencies(repo, downloads, stage, python_exe))
        meta["torch_gate"] = gate_torch(python_exe)
        meta.update(stage_payload(repo, stage, python_exe, out_dir / "_payload", args.reuse_payload))
        meta.update(stage_llamacpp(downloads, stage))
        meta.update(stage_llamacpp_quantize(downloads, stage))
        stage_launcher(stage, downloads)
        stage_assets(repo, stage)
        stage_notices(repo, stage, downloads)

    staged_python = stage / "App" / "python" / "python.exe"
    meta["gguf_gate"] = gate_gguf_export(staged_python, stage, downloads, out_dir / "_gguf_gate")

    total_bytes = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file())
    check_size(total_bytes)
    deepest, deepest_total = check_max_path(stage, msix_ver)
    info = write_manifest_and_info(stage, msix_ver, meta, total_bytes, deepest, deepest_total)
    (out_dir / "build-info.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    _log("size report:")
    for child in sorted(stage.iterdir()):
        if child.is_dir():
            size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
            print(f"  {child.name + '/':<28} {size / 1024**2:>9.0f} MiB", flush=True)
        else:
            size = child.stat().st_size
            if size > 1 << 20:
                print(f"  {child.name:<28} {size / 1024**2:>9.0f} MiB", flush=True)
    print(f"  {'TOTAL':<28} {total_bytes / 1024**3:>9.2f} GiB (cap {SIZE_CAP_BYTES / 1024**3:.0f} GiB)", flush=True)
    print(f"  MAX_PATH deepest: {deepest_total} chars incl. install prefix (limit {MAX_PATH_LIMIT})", flush=True)

    msix = pack(stage, out_dir, f"backpropagate_{msix_ver}_x64.msix")
    if args.sideload_test:
        sideload_sign(msix)
    _log(f"done in {(time.monotonic() - t0) / 60:.1f} min -> {msix}")
    _log("next: WACK (appcert.exe validate), sideload verification, then Partner Center upload of the UNSIGNED build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
