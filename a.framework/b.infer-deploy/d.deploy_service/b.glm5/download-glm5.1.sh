import os
from openmind_hub import snapshot_download

# token 通过环境变量传入, 不要硬编码到脚本里(避免泄露):
#   export OPENMIND_HUB_TOKEN=<你的 openmind token>
TOKEN = os.environ.get("OPENMIND_HUB_TOKEN", "")

if not TOKEN:
    raise SystemExit("请先 export OPENMIND_HUB_TOKEN=<你的 openmind token> 再运行")

snapshot_download(
    "Eco-Tech/GLM-5.1-w4a8",
    token=TOKEN,
    repo_type="model",
)
