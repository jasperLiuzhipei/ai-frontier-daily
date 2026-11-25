# AI Daily — 自迭代 AI 前沿与硬核技术日报

一套**零第三方依赖（纯 Python 3.10+ 标准库）**的每日 AI 前沿情报引擎：面向**AI 产品洞察 × 大模型训推与架构 × 硬核极客品味**，每天自动扫描 20 个高信噪比一手源，经过**原始时效穿透、7 天跨日事件去重、正文与 HN 热评深抓**后，由 **DeepSeek V4（OpenAI 兼容 API）** 执行**去分类化 OR-Gate 选品 + 动态主题分组**，以**飞书原生交互卡片**零跳转推送，并支持**模型 API 与信源/偏好双层自主迭代**。

---

## 系统流水线（v3 架构）

```text
[20 个一手信源并发抓取]
        │
        ▼
[Stage 1: 时效穿透 + 7天跨日硬去重 + 热度均衡初筛]
  • URL / 规范标题 / GitHub 仓库 / (产品名, 版本号) 跨日拦截（data/sent_history.json）
  • URL 路径日期（/2026/08/）+ arXiv:YYMM + 版本快照戳（-1017）原始时间穿透
        │
        ▼
[Stage 1.5: Top 18 正文深抓 + HN 一线热评 + 元数据发布时间核验]
  • 抓取原文正文（~2400 字）、GitHub/HF 仓库与安装命令锚点（uv / pip / brew）
  • 解析 HTML <meta article:published_time> / JSON-LD datePublished 拦截翻新旧文
  • 拉取 HN 前 4 条开发者热评，捕获真实踩坑与社区反应
        │
        ▼
[Stage 2: DeepSeek 精排（OR-Gate 单峰打分 + 动态分组 + 双轨深度抽取）]
  • 注入近 7 天已发事件清单，防换壳重播，仅重大变故以【追踪进展】放行
  • 🔬 技术 / 🎲 趣味 / 💡 冲击 三维独立打分 → 0.6×最高项 + 0.4×其余均值
  • 动态归纳 2~4 个当日主题分组，每条输出 核心要点 + 🔬 技术/架构 + 💡 启发 + ⚡ 极客入口
        │
        ▼
[推送 & 持久化]
  • 飞书原生交互卡片直读（自建应用 API uuid 幂等主推，Webhook 兼容 + 30KB 保护）
  • 归档 output/YYYY-MM-DD.md + 更新 data/sent_history.json + SQLite 命中率遥测
```

---

## 核心特性

### 1. 20 个高信噪比一手信源（`config/sources.json`）

| 方向 | 信源 | 说明 |
| :--- | :--- | :--- |
| **高密聚合门户（2）** | **AI Hot** (`aihot.news`)、**Smol.ai AI News** | 中英双语热点 API（带备用域名与 Markdown 回退）+ 海外工程师每日精华 |
| **Agent Harness（5）** | Google Antigravity Changelog、OpenAI Codex Releases、Anthropic Claude Code Releases、Anthropic Engineering、Harness 定向追踪 | Coding Agent 三巨头一手更新雷达与上下文工程实践 |
| **模型与底层动态（4）** | DeepSeek API Updates、Google DeepMind Blog、DeepSeek 优化定向追踪、Seedance / 字节 Seed 定向追踪 | 一线模型发布、API 能力边界与成本变化（定向源 `when:2d`） |
| **训推与硬核系统（5）** | vLLM Releases、SGLang Releases、Interconnects、Hugging Face Daily Papers、**Lobste.rs AI** | 推理引擎、后训练/RL、精选论文与底层系统工程社区 |
| **极客造物与实测（4）** | Simon Willison、Hacker News AI 热议、**Show HN（AI）**、**HF Trending Spaces** | 一线祛魅实测、脑洞开源造物、交互 Demo 与安全/翻车预警 |

### 2. 打破分类茧房 + 严格防旧防重
- **7 天跨日事件记忆（`data/sent_history.json`）**：轻量 JSON 随 Git 同步（本机与云端共享记忆），实现 URL、GitHub 仓库、`(产品, 版本)` 硬拦截 + LLM 语义防重播。
- **原始事件时间穿透**：区分“媒体/推主发帖时间”和“底层事件真实发生时间”，自动拦掉 3 个月前翻新旧文与几周前的 arXiv 旧论文科普。
- **OR-Gate 无茧房选品**：不按固定槽位凑水文，凡是 AI 相关且在**技术硬核 / 趣味脑洞 / 行业冲击**任一维度出彩均可高分入选；常规小版本更新单日最多 2 条。

### 3. 双层“自主迭代”能力
- **模型 API 层自主迭代（`models sync`）**：基于标准 OpenAI 兼容协议封装（默认使用 DeepSeek 官方 `deepseek-flash`，内置 `deepseek-v4-pro` 备选与思考模式 token 耗尽自动重试），支持自动探测 `/models` 新版本、探活测速并在备份后升级主模型。
- **日报系统自进化（`evolve` + `feedback`）**：SQLite 持续追踪各信源抓取成功率、入选命中率与新兴技术信号（`emerging_signals`），结合你的阅读反馈自动生成配置演进提案，带时间戳备份（`config/history/`）并支持随时一键 `rollback`。

### 4. 飞书自建应用消息 API（幂等防重主推）+ Webhook 兼容
- **飞书自建应用消息 API（`im/v1/messages`）**：按「日报日期 + 渠道 + 成稿哈希」生成固定 `uuid`，在飞书 1 小时去重窗口内支持超时与回执失败安全补发，同时做到 **0 漏发 + 0 重发**；并支持 `python3 run.py feishu-chats` 一键查群 ID。
- **Webhook 兼容**：如仅配置 `FEISHU_WEBHOOK_URL` 仍自动兼容（关闭底层读超时盲目重试，错误按可重试/永久失败/结果未知分类）。

---

## 快速上手（3 分钟跑通）

### 1. 静默配置本地密钥（推荐）
无需手动拷贝，直接运行静默录入命令（输入密钥时不回显，自动设 `.env` 权限 `600` 且已写入 `.gitignore`，并自动拉取机器人所在群聊选择 `FEISHU_CHAT_ID`）：

```bash
python3 run.py setup
```

或手动复制编辑 `.env`（参考 `.env.example`）：
- `DEEPSEEK_API_KEY`：DeepSeek API Key（默认走 `https://api.deepseek.com/v1`，中转站可改 `LLM_BASE_URL`）
- `FEISHU_APP_ID` / `FEISHU_APP_SECRET` / `FEISHU_CHAT_ID`：飞书自建应用凭证与目标群 ID（可用 `python3 run.py feishu-chats` 查询）
- `HEALTHCHECK_PING_URL`（可选但推荐）：外部死信监控（如 Healthchecks.io），仅在飞书确认送达且回执保存后上报成功

### 2. 测试推送与本地运行

```bash
# 1) 查询机器人已加入的飞书群 chat_id
python3 run.py feishu-chats

# 2) 发送一条飞书连通性测试卡片
python3 run.py test-notify

# 3) 本地试跑（不推送、不写去重库，生成 output/YYYY-MM-DD.md 预览）
python3 run.py run --dry-run

# 4) 正式运行并推送到飞书群
python3 run.py run

# 5) 查看最近 14 天交付回执与无人值守观测指标
python3 run.py status --days 14
```

---

## 常用运维与自主迭代命令

```bash
# 查看当前模型状态 / 探测并升级最新可用模型
python3 run.py models status
python3 run.py models sync
python3 run.py models sync --apply

# 管理信源 & 查看历史入选命中率
python3 run.py sources list
python3 run.py sources stats
python3 run.py sources add --id my_blog --name "某硬核博客" --url "https://example.com/rss.xml" --category systems
python3 run.py sources toggle <source_id>

# 记录阅读偏好反馈，驱动系统自进化
python3 run.py feedback \
  --topic "Looped Transformer" --like "AI安全攻防" --dislike "自媒体凑数横评" \
  --note "多保留有真实消融数据和开源代码的文章"

# 生成 / 应用自主迭代提案，或一键回滚任意配置
python3 run.py evolve
python3 run.py evolve --apply
python3 run.py evolve --history
python3 run.py rollback profile   # 支持 llm / profile / sources / prompts
```

---

## 每日全自动运行（GitHub Actions，零服务器成本）

已内置两套职责解耦的工作流：
1. **`.github/workflows/daily_digest.yml`**（每日发送关键路径，`TZ: Asia/Shanghai`，并发组 `ai-daily-digest` 且 `cancel-in-progress: false`）：
   - **10:23 主跑（`23 2 * * *`）**：生成固定成稿（Markdown + 飞书卡片）并写入独立状态分支 `delivery-state` → 携带确定性 `uuid` 推送飞书 → 写回执 → 仅确认送达后提交 `output/YYYY-MM-DD.md` 和 `data/sent_history.json` 到 `main` → 上报外部死信监控。若主备模型全挂，本轮报错退出交由 10:47 重试完整 LLM。
   - **10:47 兜底（`47 2 * * *`）**：读取 `delivery-state` 最新回执——已确认送达则自动跳过；首推失败/回执未落盘则**直接复用成稿 + 同一个 `uuid` 补发**（零重新抓源与 LLM 消耗，飞书 1 小时窗口自动去重）；若主备模型仍全挂则发送标明「⚠️ LLM 降级短报」的规则保底短报。
   - **所有触发（定时 / `repository_dispatch` / `workflow_dispatch`）共用同一并发组与幂等回执检查**，任意关键失败均返回非零退出码。
2. **`.github/workflows/weekly_evolve.yml`**（每周日 15:30 独立运行，不碰线上配置）：
   - 独立执行模型探测（`models sync`）与自演进提案生成（`evolve`），结果保存为 Workflow Artifact 与 Job Summary，审核后按需手动 `--apply`。

在你的 GitHub 仓库 `Settings → Secrets and variables → Actions` 配置以下 Secrets 即可生效（未配置必需 Secrets 时两个工作流会自动跳过，不会报错或创建 `delivery-state` 分支）：
- `DEEPSEEK_API_KEY`（必填）
- `FEISHU_APP_ID` / `FEISHU_APP_SECRET` / `FEISHU_CHAT_ID`（推荐，飞书自建应用 API + `uuid` 幂等）
- `NOTIFY_CHANNEL`（可选，默认 `auto` 优先飞书应用 API，可设为 `feishu` / `feishu-app` / `feishu-webhook`）
- `HEALTHCHECK_PING_URL`（强烈推荐，如 Healthchecks.io 免费版，配 11:15 截止窗口 + 邮件告警）
- `FEISHU_WEBHOOK_URL` / `FEISHU_WEBHOOK_SECRET` / `LLM_BASE_URL` / `LLM_MODEL`（按需选填）

---

## 项目结构

```text
AI_Daily/
├── run.py                               # 零配置根入口（python3 run.py ...）
├── scripts/sync_state_branch.sh         # 独立状态分支 delivery-state 的 worktree 同步与重试推送
├── config/
│   ├── llm.json                         # DeepSeek V4 主备模型、thinking 参数与自动迭代策略
│   ├── profile.json                     # 读者画像、时间窗、正文深抓与屏蔽词配置
│   ├── sources.json                     # 20 个分层一手信源配置
│   └── prompts.json                     # v3 无茧房 OR-Gate 选品 + 动态分组 + 自进化 Prompt
├── data/
│   ├── sent_history.json                # 7~14 天跨日事件记忆库（仅确认送达后更新）
│   └── model_registry.json              # 模型探活与版本同步记录
├── output/                              # 每日 Markdown 日报归档（仅确认送达后落盘）
├── src/ai_daily/
│   ├── fetchers/                        # RSS/Atom, AI Hot, HN/Show HN, HF Papers/Spaces, GitHub, WebWatch
│   ├── llm/                             # OpenAI 兼容客户端（坏 JSON 自动切备模 + 思考模式降级）+ 模型注册表
│   ├── pipeline/                        # 时效穿透过滤、深抓、OR-Gate 精选与降级短报、分阶段编排
│   ├── notifiers/                       # 飞书自建应用消息 API（uuid 幂等）+ Webhook 兼容 + 错误分级
│   ├── delivery.py                      # 固定成稿（draft.json）与按渠道回执（receipt.json）状态机
│   ├── monitor.py                       # 外部死信监控（HEALTHCHECK_PING_URL）上报
│   ├── evolution/                       # 信源健康与反馈驱动的自主迭代引擎
│   ├── history.py                       # 跨日事件指纹提取与记忆库管理
│   └── storage.py                       # SQLite 去重与信源命中率遥测
├── .github/workflows/daily_digest.yml   # 北京时间 10:23 + 10:47 并发互斥双保险云端工作流
├── .github/workflows/weekly_evolve.yml  # 每周独立产出自演进与模型探测提案（不改线上配置）
└── tests/                               # 35 项零网络单元、端到端与故障注入测试
```

---

## 运行测试

```bash
python3 -m unittest discover -s tests -v
```

---

## License

[MIT License](LICENSE)
