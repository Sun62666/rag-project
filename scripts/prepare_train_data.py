import json
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent.parent


def prepare_from_chunks(output_file: str = None):
    from src.finetune import OpsDataBuilder

    pdf_path = str(_BASE_DIR / "data" / "文档2.pdf")
    if not output_file:
        # 优先使用 data/prepared/ 目录
        output_file = str(_BASE_DIR / "data" / "prepared" / "ops_from_chunks.json")

    if not Path(pdf_path).exists():
        logger.error(f"PDF 不存在: {pdf_path}")
        return

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    samples = OpsDataBuilder.from_retriever_chunks(pdf_path, output_file)
    logger.info(f"[chunks] 生成 {len(samples)} 条训练数据 → {output_file}")
    return samples


def prepare_from_manual(output_file: str = None):
    from src.finetune import OpsDataBuilder

    manual_file = str(_BASE_DIR / "data" / "prepared" / "manual_qa.json")
    if not output_file:
        output_file = str(_BASE_DIR / "data" / "prepared" / "ops_from_manual.json")

    if not Path(manual_file).exists():
        template = [
            {"query": "Redis内存溢出如何排查？", "answer": "【故障现象】Redis进程占用内存持续增长\n【排查】redis-cli info memory\n【修复】config set maxmemory-policy allkeys-lru", "source": "manual"},
            {"query": "MySQL主从延迟怎么处理？", "answer": "【故障现象】从库Seconds_Behind_Master持续增大\n【排查】show slave status\\G\n【修复】set global slave_parallel_workers=4", "source": "manual"},
        ]
        Path(manual_file).parent.mkdir(parents=True, exist_ok=True)
        with open(manual_file, "w", encoding="utf-8") as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        logger.info(f"已生成手动标注模板 → {manual_file}")
        logger.info("请编辑该文件补充更多 Q&A 对，然后重新运行")
        return

    samples = OpsDataBuilder.from_manual_pairs(manual_file, output_file)
    logger.info(f"[manual] 生成 {len(samples)} 条训练数据 → {output_file}")
    return samples


def prepare_synthetic(output_file: str = None, num_samples: int = 200):
    from src.finetune import OpsDataBuilder

    if not output_file:
        output_file = str(_BASE_DIR / "data" / "prepared" / "ops_synthetic.json")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    samples = OpsDataBuilder.generate_synthetic_data(output_file, num_samples)
    logger.info(f"[synthetic] 生成 {len(samples)} 条训练数据 → {output_file}")
    return samples


def prepare_rerank_data(output_file: str = None):
    rerank_data = [
        {
            "query": "Redis内存溢出如何排查？",
            "positive_contexts": [
                "Redis内存溢出通常由未设置maxmemory或大Key堆积导致，可通过redis-cli info memory查看内存使用情况",
                "Redis OOM排查步骤：1.检查maxmemory配置 2.使用redis-cli --bigkeys扫描大Key 3.设置淘汰策略",
            ],
            "negative_contexts": [
                "MySQL连接超时可能是wait_timeout配置过小导致，可通过show processlist查看",
                "Nginx 502错误通常表示后端服务不可用，需要检查upstream配置",
            ],
        },
        {
            "query": "MySQL主从同步延迟怎么处理？",
            "positive_contexts": [
                "MySQL主从延迟排查：show slave status\\G查看Seconds_Behind_Master，开启多线程复制可缓解",
                "主从延迟修复：1.开启并行复制slave_parallel_workers 2.拆分大事务 3.升级从库硬件",
            ],
            "negative_contexts": [
                "Redis集群脑裂问题需要配置min-replicas-to-write参数",
                "Docker容器网络不通可能是iptables规则冲突导致",
            ],
        },
        {
            "query": "Nginx 502 Bad Gateway如何排查？",
            "positive_contexts": [
                "Nginx 502错误表示后端服务不可用，排查步骤：1.检查后端服务状态 2.查看error.log 3.确认upstream配置",
                "502错误修复：调整proxy_read_timeout、检查后端端口、配置健康检查max_fails参数",
            ],
            "negative_contexts": [
                "Kubernetes Pod CrashLoopBackOff需要查看kubectl logs定位崩溃原因",
                "Elasticsearch集群变红表示主分片不可用，需要检查节点状态",
            ],
        },
        {
            "query": "CPU使用率100%怎么排查？",
            "positive_contexts": [
                "CPU使用率100%排查：使用top -c定位高CPU进程，top -H -p PID分析线程，vmstat查看系统级CPU状态",
                "CPU飙高常见原因：进程死循环、突发流量、定时任务执行、内存不足导致频繁swap",
            ],
            "negative_contexts": [
                "Redis持久化fork导致COW内存翻倍，需要优化持久化配置",
                "Zookeeper连接超时可能是session timeout配置过小",
            ],
        },
        {
            "query": "Kubernetes Pod CrashLoopBackOff如何解决？",
            "positive_contexts": [
                "CrashLoopBackOff排查：kubectl describe pod查看事件，kubectl logs --previous查看上次崩溃日志",
                "Pod崩溃常见原因：OOMKilled、启动命令错误、配置缺失、健康检查配置不当",
            ],
            "negative_contexts": [
                "Linux磁盘满可通过du -sh /* | sort -rh定位大目录",
                "Kafka消费积压需要增加消费者实例数或优化消费逻辑",
            ],
        },
        {
            "query": "Linux磁盘空间满如何清理？",
            "positive_contexts": [
                "磁盘满排查：df -h查看分区使用率，du -sh定位大目录，lsof | grep deleted查找已删除但被占用的文件",
                "磁盘清理步骤：1.清理过期日志 2.删除临时文件 3.释放已删除文件 4.docker system prune清理Docker",
            ],
            "negative_contexts": [
                "Docker容器网络不通需要检查网桥配置和iptables规则",
                "Redis内存溢出可通过设置maxmemory和淘汰策略解决",
            ],
        },
        {
            "query": "Docker容器网络不通怎么排查？",
            "positive_contexts": [
                "Docker网络排查：docker network ls查看网络列表，docker network inspect检查配置，docker exec ping测试连通性",
                "容器网络修复：1.加入同一网络 2.重建网桥 3.重启Docker恢复iptables 4.指定DNS服务器",
            ],
            "negative_contexts": [
                "MySQL主从延迟可通过开启多线程复制缓解",
                "Nginx 502错误需要检查后端服务状态",
            ],
        },
        {
            "query": "Elasticsearch集群变红怎么处理？",
            "positive_contexts": [
                "ES集群变红表示主分片不可用，排查：_cluster/health查看状态，_cat/shards查找UNASSIGNED分片",
                "ES红色状态修复：1.重启宕机节点 2.清理磁盘 3.手动分配分片_cluster/reroute 4.增加副本数",
            ],
            "negative_contexts": [
                "CPU使用率100%需要用top定位高CPU进程",
                "Zookeeper连接超时可能是JVM GC停顿导致",
            ],
        },
        {
            "query": "Kafka消费积压如何处理？",
            "positive_contexts": [
                "Kafka消费积压排查：kafka-consumer-groups.sh --describe查看Lag，确认消费者数量和分区数匹配",
                "消费积压修复：1.增加消费者实例 2.优化消费逻辑 3.扩容分区 4.调整max.poll.interval.ms",
            ],
            "negative_contexts": [
                "Linux磁盘满需要用df -h和du -sh定位大文件",
                "Redis OOM可通过设置maxmemory-policy解决",
            ],
        },
        {
            "query": "Zookeeper连接超时怎么处理？",
            "positive_contexts": [
                "ZK连接超时排查：echo ruok | nc测试连通性，echo mntr查看监控指标，检查session timeout配置",
                "ZK超时修复：1.增大session timeout 2.优化JVM堆内存和GC 3.检查网络延迟 4.扩容ZK集群",
            ],
            "negative_contexts": [
                "Nginx 502错误需要检查后端服务是否正常运行",
                "Kubernetes Pod崩溃需要查看kubectl logs",
            ],
        },
    ]

    if not output_file:
        output_file = str(_BASE_DIR / "data" / "prepared" / "reranker_train.json")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rerank_data, f, ensure_ascii=False, indent=2)

    total_pos = sum(len(d["positive_contexts"]) for d in rerank_data)
    total_neg = sum(len(d["negative_contexts"]) for d in rerank_data)
    logger.info(f"[rerank] 生成 {len(rerank_data)} 组数据 (正例:{total_pos}, 负例:{total_neg}) → {output_file}")
    return rerank_data


def prepare_from_crawled(output_file: str = None, max_per_source: int = 100):
    crawled_dir = _BASE_DIR / "data" / "crawled"
    if not crawled_dir.exists():
        logger.error(f"爬取数据目录不存在: {crawled_dir}")
        return

    if not output_file:
        output_file = str(_BASE_DIR / "data" / "prepared" / "ops_from_crawled.json")

    samples = []
    for source_dir in sorted(crawled_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        source_name = source_dir.name
        md_files = list(source_dir.glob("*.md"))
        logger.info(f"  处理 {source_name}: {len(md_files)} 个文件")

        count = 0
        for md_file in md_files:
            if count >= max_per_source:
                break
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()

                lines = content.split("\n")
                clean_lines = []
                for line in lines:
                    if line.startswith("# 来源:") or line.startswith("# 爬取时间:"):
                        continue
                    clean_lines.append(line)
                content = "\n".join(clean_lines).strip()

                if len(content) < 100:
                    continue

                chunks = _split_to_chunks(content, max_len=800)
                for chunk in chunks:
                    qa = _generate_qa_from_chunk(chunk, source_name)
                    if qa:
                        samples.append(qa)
                        count += 1
            except Exception as e:
                logger.warning(f"  处理失败 {md_file.name}: {e}")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    logger.info(f"[crawled] 生成 {len(samples)} 条训练数据 -> {output_file}")
    return samples


def _split_to_chunks(text: str, max_len: int = 800) -> List[str]:
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) > max_len and current:
            chunks.append(current.strip())
            current = line + "\n"
        else:
            current += line + "\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _generate_qa_from_chunk(chunk: str, source_name: str) -> Optional[Dict]:
    headings = re.findall(r'^#+\s+(.+)', chunk, re.MULTILINE)
    title = headings[0].strip() if headings else source_name

    if len(chunk) < 50:
        return None

    query = f"请介绍{title}的相关内容"
    return {
        "input": query,
        "output": chunk[:600],
        "source": source_name,
    }


def merge_all_sources(output_file: str = None):
    if not output_file:
        output_file = str(_BASE_DIR / "data" / "prepared" / "ops_train.json")

    all_samples = []
    data_dir = _BASE_DIR / "data" / "prepared"

    for name in ["ops_from_crawled.json", "ops_from_chunks.json", "ops_from_manual.json", "ops_synthetic.json"]:
        fpath = data_dir / name
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_samples.extend(data)
            logger.info(f"  合并 {name}: {len(data)} 条")

    seen = set()
    unique = []
    for s in all_samples:
        key = f"{s['input']}|{s['output'][:100]}"
        if key not in seen:
            seen.add(key)
            unique.append(s)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    logger.info(f"合并去重后共 {len(unique)} 条训练数据 → {output_file}")
    return unique


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        prepare_from_crawled()
        prepare_from_chunks()
        prepare_from_manual()
        prepare_synthetic()
        prepare_rerank_data()
        merge_all_sources()
    elif len(sys.argv) > 1 and sys.argv[1] == "--crawled":
        prepare_from_crawled()
    elif len(sys.argv) > 1 and sys.argv[1] == "--rerank":
        prepare_rerank_data()
    elif len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_all_sources()
    else:
        print("""
SmartOps 训练数据准备工具
=========================

用法:
  python prepare_train_data.py --all      # 准备所有数据源并合并
  python prepare_train_data.py --crawled  # 从爬取数据生成训练样本
  python prepare_train_data.py --rerank   # 仅准备 Reranker 训练数据
  python prepare_train_data.py --merge    # 合并已有数据源

数据源:
  1. crawled   - 从爬取的中文文档自动生成 (推荐优先)
  2. chunks    - 从 PDF 文档切片自动生成
  3. manual    - 从手动标注 Q&A 对生成
  4. synthetic - 从运维模板合成数据
  5. rerank    - Reranker 正负例标注数据

生成文件:
  data/prepared/
  +-- ops_from_crawled.json   # 来自爬取文档
  +-- ops_from_chunks.json    # 来自PDF切片
  +-- ops_from_manual.json    # 来自手动标注
  +-- ops_synthetic.json      # 合成数据
  +-- reranker_train.json     # Reranker训练数据
  +-- ops_train.json          # 合并后的最终训练集
""")
