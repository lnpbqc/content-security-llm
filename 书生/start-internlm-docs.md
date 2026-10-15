# 书生启动说明

前置安装与模型下载见 [部署手册](./InternLM-A100部署手册.md)。激活 `internlm` 环境，在 `~/internlm/` 执行：

```bash
./start-internlm.sh start
./start-internlm.sh status
./start-internlm.sh log       # Ctrl+C 只退出日志
./start-internlm.sh stop
./start-internlm.sh restart
./start-internlm.sh fg        # 前台运行
```

默认：`GPU=0 PORT=1221 SESSION_LEN=32768 CACHE=0.2 BATCH=4 MAXCONC=4`。

同卡先起书生，待 1221 接口就绪再起 Qwen。Key 文件为 `~/internlm/api.key`，模型目录为 `/data/models/internlm3-8b-instruct`。

临时覆盖参数并重启：

```bash
SESSION_LEN=32768 CACHE=0.2 BATCH=4 MAXCONC=4 ./start-internlm.sh restart
```

长文本单请求选项，先验证最长业务输入与实际显存：

```bash
SESSION_LEN=131072 BATCH=1 MAXCONC=1 ./start-internlm.sh restart
```

32K 包含输入与输出，业务接口上限不变。后续启动或重启时继续传入所需覆盖值；`stop` 不会保存参数。显存通过 `status` 或 `nvidia-smi` 检查，不按固定占用估算。

结构化业务需由调用方发送对应 `response_format`，并保留返回后校验；普通文本不加 Schema。仅修改文档或重启模型不会改变业务请求。
