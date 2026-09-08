import yaml
from pathlib import Path

# 用"当前文件所在的文件夹"定位配置文件，而不是写死带空格的相对路径
# 这样无论你在哪个目录运行，都能找到 config.yaml
config_path = Path(__file__).parent / "config.yaml"

with open(config_path, encoding="utf-8") as f:
    data = yaml.safe_load(f)

data["temperature"] = 0.3

with open(config_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
