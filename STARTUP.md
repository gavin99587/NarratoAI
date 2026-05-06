# NarratoAI 手动启动

在项目根目录执行：

```bash
cd /Users/zhanghuiwen/codex_space/NarratoAI
source .venv/bin/activate
HOME=/Users/zhanghuiwen/codex_space/NarratoAI/.home streamlit run webui.py --server.maxUploadSize=2048 --server.port=8502
```

如需使用 `8501` 端口，把最后的 `--server.port=8502` 改为 `--server.port=8501`。
