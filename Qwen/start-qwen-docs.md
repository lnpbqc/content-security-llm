# Qwen 启动说明

前置安装与模型下载见 [部署手册](./Qwen3-8B-A100部署手册.md)。激活 `qwen` 环境，在 `~/qwen/` 执行：

```bash
./start-qwen.sh start
./start-qwen.sh status
./start-qwen.sh log       # Ctrl+C 只退出日志
./start-qwen.sh stop
./start-qwen.sh restart
./start-qwen.sh fg        # 前台运行
```

默认：`GPU=0 PORT=1222 SESSION_LEN=32768 CACHE=0.3 BATCH=4 MAXCONC=4`。

同卡先起书生，待 1221 接口就绪再起 Qwen。Key 文件为 `~/qwen/api.key`，模型目录为 `/data/models/Qwen3-8B`。

临时覆盖参数并重启：

```bash
SESSION_LEN=32768 CACHE=0.3 BATCH=4 MAXCONC=4 ./start-qwen.sh restart
```

脚本默认解析器为 `default`。若安装版本确认支持 `qwen3`，可覆盖并重启：

```bash
REASONING_PARSER=qwen3 ./start-qwen.sh restart
```

后续启动或重启时继续传入所需覆盖值；`stop` 不会保存参数。解析器不会关闭思考；非思考结构化请求需由业务调用方发送 `chat_template_kwargs.enable_thinking=false` 和对应 Schema，见 [部署手册](./Qwen3-8B-A100部署手册.md#5-结构化业务与思考开关)。

32K 包含输入与输出，业务接口上限不变。显存通过 `status` 或 `nvidia-smi` 检查，不保证四条满长请求同时执行。遇到空正文先核对请求及模型日志，不用思考字段代替结果，不自动放宽校验。
