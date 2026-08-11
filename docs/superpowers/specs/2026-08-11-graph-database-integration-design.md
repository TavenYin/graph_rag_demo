# 图数据库接入设计

## 目标

在现有 PostgreSQL/pgvector RAG 基线旁接入本地 Neo4j，用于受控实体关系、实体链接和 1～2 跳关系扩展。

Neo4j 不保存独立的知识正文，也不直接作为回答依据。它只负责发现与问题相关的实体、关系路径和来源 chunk；回答仍使用 PostgreSQL 中的原始 `kb_chunk` 内容。

## 已有基础

当前链路为：文档清洗、按 token 分块、批量向量化、写入 PostgreSQL/pgvector、查询扩写、向量检索与中文全文检索、一次全局 RRF、受 token 预算限制的 XML 知识上下文、LLM 回答。

`kb_chunk.id` 是图谱中实体和关系回溯原始证据的稳定标识。

## 范围

第一期包含：

- 本地 Neo4j 连接与生命周期管理。
- LLM 对每个主 chunk 抽取受控实体和关系，并使用相邻 chunk 辅助理解。
- 实体归一化、别名与来源 chunk 关联。
- 实体搜索：精确名称、别名、Neo4j 全文索引和实体向量候选。
- 命中实体后按允许的关系进行 1～2 跳扩展。
- 依据图中 `chunk_id` 回取 PostgreSQL 原文，并参与现有的一次全局 RRF。

第一期不包含：

- 任意关系类型的自由写入。
- 三跳及以上的默认遍历。
- 用 Neo4j 替换 PostgreSQL、pgvector 或全文检索。
- 复杂的查询规划、意图路由、图谱后台或异步任务队列。

## 受控 Schema 配置

图谱 Schema 存储在 PostgreSQL，而不是硬编码在应用或存储在 Neo4j。Neo4j 只保存实体和关系实例；PostgreSQL 是实体类型、关系类型及其约束的唯一来源。

第一版使用两张配置表：

```text
graph_entity_type
- code
- description
- is_active

graph_relation_type
- code
- description
- source_types JSONB
- target_types JSONB
- is_active
```

服务在抽取前读取启用的配置，动态构造提示词中的实体和关系枚举。提示词本身保持通用：它只说明抽取 JSON 格式、证据要求和“只能从给定 Schema 中选择”；实体与关系的具体名称、说明、方向和适用类型由配置数据提供。

模型输出后，服务端按同一份配置重新校验实体类型、关系类型和起终点类型。配置表不提供第一期管理 API；初始 Schema 通过数据库初始化脚本写入，后续由数据库数据变更维护。

### 实体类型

下列记录是第一期写入 `graph_entity_type` 的初始数据：

| 类型 | 含义 |
| --- | --- |
| `Feature` | 功能、玩法 |
| `Quest` | 任务 |
| `NPC` | 角色 |
| `Location` | 地点 |
| `Item` | 道具 |
| `Level` | 等级 |

### 关系类型

下列记录是第一期写入 `graph_relation_type` 的初始数据：

| 类型 | 起点 | 终点 | 语义 |
| --- | --- | --- | --- |
| `UNLOCKED_BY` | `Feature`、`Location` | `Quest`、`Level`、`Item` | 起点由终点解锁 |
| `REQUIRES_QUEST` | `Feature`、`Quest`、`Location` | `Quest` | 起点要求先完成终点任务 |
| `REQUIRES_LEVEL` | `Feature`、`Quest`、`Location` | `Level` | 起点要求达到终点等级 |
| `TRIGGERED_BY` | `Quest` | `NPC` | 任务由 NPC 触发 |
| `LOCATED_IN` | `Feature`、`Quest`、`NPC`、`Item` | `Location` | 起点位于终点地点 |
| `REWARDS` | `Feature`、`Quest` | `Item` | 起点奖励终点道具 |

LLM 只能从运行时读取的启用实体和关系枚举中选择。不能归类的信息可以输出 `UNKNOWN` 及建议类型或关系，但仅记录用于后续治理，不写入正式图谱。

## 图模型

第一版使用一个节点标签和一个边类型，减少数据库结构和查询分支：

```text
(:Entity {
  id,
  name,
  type,
  aliases,
  embedding,
  chunk_ids
})

(:Entity)-[:RELATION {
  type,
  evidence,
  chunk_id,
  confidence
}]->(:Entity)
```

- `Entity.id`：应用生成的稳定 ID，供写入幂等和查询使用。
- `name`：归一化后的标准名称。
- `type`：受控实体类型。
- `aliases`：可用于名称匹配的别名集合。
- `embedding`：实体名称和别名生成的向量，用于候选实体召回。
- `chunk_ids`：出现该实体的 PostgreSQL chunk ID 集合。
- `RELATION.type`：受控关系类型。
- `RELATION.evidence`：支持关系的原文摘录。
- `RELATION.chunk_id`：抽取该关系的来源 chunk。
- `RELATION.confidence`：抽取置信度，仅供观测和后续筛选。

边的 `type` 不由模型自由命名；它是上方关系枚举中的一个值。关系方向也由 Schema 固定，例如 `Feature -[UNLOCKED_BY]-> Quest` 表示“功能由任务解锁”。

## 入库与抽取

```text
文档上传
→ token 分块并写入 PostgreSQL
→ 为每个主 chunk 组成相邻上下文窗口
→ 对该窗口调用 LLM 抽取
→ 校验 JSON、实体类型、关系类型、方向与 evidence
→ 归一化实体名称，合并同名同类型实体
→ 写入 Neo4j 节点、关系和 chunk 来源
```

抽取的写入单位是一个主 chunk，而不是多个任意拼接的 chunk。发送给模型的窗口包含前一个、当前主 chunk 和后一个 chunk：

```text
[上文，仅辅助理解] chunk_9
[当前主 chunk，允许作为证据] chunk_10
[下文，仅辅助理解] chunk_11
```

相邻 chunk 只用于处理代词、省略主语和分块截断。输出实体和关系的 `evidence` 必须能够在主 chunk 中匹配，图写入只关联主 chunk ID；若一个关系必须同时依赖多个 chunk 才能成立，第一期不写入该关系。

抽取输出包含临时实体 ID，避免同一主 chunk 内关系依赖自然语言名称：

```json
{
  "entities": [
    {"id": "e1", "name": "梦境功能", "type": "Feature", "aliases": ["梦境"]},
    {"id": "e2", "name": "星夜之门", "type": "Quest", "aliases": []}
  ],
  "relations": [
    {
      "source_id": "e1",
      "type": "UNLOCKED_BY",
      "target_id": "e2",
      "evidence": "完成星夜之门后开启梦境功能",
      "confidence": 0.95
    }
  ],
  "unknowns": []
}
```

应用校验 `evidence` 必须可在主 chunk 中找到或能被规范化后匹配；不满足时拒绝该实体或关系写入。

## 检索设计

现有查询扩写、向量检索和全文检索保持不变。图检索增加为一个候选来源，最终仍只做一次全局 RRF。

```text
用户问题
→ 查询扩写
→ 向量检索与全文检索
→ LLM 提取待链接实体文本
→ 实体搜索
  1. 标准名称精确匹配
  2. aliases 匹配
  3. Neo4j 全文索引
  4. 实体向量 Top-K
→ 选择候选实体
→ 按受控关系扩展 1～2 跳
→ 收集关系和节点的 chunk_id
→ PostgreSQL 回取原始 chunk
→ 与现有候选列表合并、去重、一次全局 RRF
→ XML knowledge
→ LLM JSON 回答
```

第一版默认最多两跳。图遍历必须限定允许的关系类型和结果数量，防止无关上下文快速扩张。

实体链接的优先级是精确名称和别名。若命中多个候选，再由实体类型、完整问题语义和实体向量分数排序；无法达到置信阈值时不做图扩展，保留原有 RAG 回答。

## 提示词与模型职责

模型不生成 Cypher，也不决定数据库的真实边名称。模型只负责：

- 在入库时从运行时提供的受控 Schema 中抽取实体和关系。
- 在查询时从问题中识别待链接实体文本。
- 在候选实体歧义时选择候选 ID 或放弃链接。

应用负责：

- 基于 PostgreSQL Schema 配置的校验与方向校验。
- 实体归一化和 Neo4j 参数化 Cypher。
- 关系跳数、结果数量与来源 chunk 限制。
- PostgreSQL 原文回取、RRF 与最终 XML 知识构造。

## 失败处理与观测

- 实体/关系抽取失败：不影响已提交的 PostgreSQL 文档；记录失败并跳过该 chunk 的图写入。
- 图数据库暂不可用：文档入库和现有 RAG 查询继续工作；图检索降级为空候选来源。
- 抽取结果不合法：记录拒绝原因，不写入 Neo4j。
- 实体链接不确定：不进行图扩展。

日志只记录 chunk ID、实体 ID、实体类型、关系类型、候选数量、跳数和分数；不记录完整原文、提示词或模型答案。

## 测试

- 单元测试：动态 Schema 提示词、抽取 JSON 校验、关系方向、主 chunk 证据约束、实体归一化、XML 证据回溯和图检索降级。
- Neo4j 集成测试：节点/边幂等写入、别名搜索、全文搜索、实体向量候选、1～2 跳扩展。
- PostgreSQL + Neo4j 链路测试：图关系返回的 chunk ID 能回取原文，并与现有候选一起在一次 RRF 中融合。
- 回归测试：Neo4j 不可用时 `/documents` 和 `/ask` 的既有向量/全文路径仍可用。

## 分阶段交付

1. Neo4j 客户端、Compose 配置、受控 Schema 与图模型。
2. PostgreSQL Schema 初始数据、动态提示词、主 chunk 加相邻上下文的 LLM 抽取、校验、归一化和图写入。
3. 实体搜索和 1～2 跳关系扩展。
4. 图候选回取原始 chunk，并并入现有全局 RRF。

每个阶段都保持现有 RAG 可独立运行，避免图层故障阻断基础检索。
