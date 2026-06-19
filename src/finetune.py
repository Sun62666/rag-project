import json
import os
import logging
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from peft import LoraConfig, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent.parent


class OpsDataBuilder:
    """运维微调数据构建器：从项目现有数据源自动生成训练集"""

    @staticmethod
    def from_retriever_chunks(
        pdf_path: str,
        output_file: str,
        max_samples: int = 500,
    ) -> List[Dict]:
        from src.retriever import OpsRetriever

        retriever = OpsRetriever(pdf_path=pdf_path)
        if not hasattr(retriever, "splits") or not retriever.splits:
            logger.error("检索器切片为空")
            return []

        samples = []
        for doc in retriever.splits[:max_samples]:
            content = doc.page_content.strip()
            if len(content) < 50:
                continue

            query = OpsDataBuilder._extract_query_from_chunk(content)
            answer = OpsDataBuilder._format_ops_answer(content)

            samples.append({
                "instruction": "你是一个运维专家，请根据知识回答运维问题。",
                "input": query,
                "output": answer,
                "source": doc.metadata.get("source", ""),
            })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        logger.info(f"从 {len(retriever.splits)} 个 chunk 生成 {len(samples)} 条训练数据")
        return samples

    @staticmethod
    def from_manual_pairs(
        pairs_file: str,
        output_file: str,
    ) -> List[Dict]:
        with open(pairs_file, "r", encoding="utf-8") as f:
            pairs = json.load(f)

        samples = []
        for item in pairs:
            samples.append({
                "instruction": "你是一个运维专家，请根据知识回答运维问题。",
                "input": item["query"],
                "output": item["answer"],
                "source": item.get("source", "manual"),
            })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        logger.info(f"从手动标注文件生成 {len(samples)} 条训练数据")
        return samples

    @staticmethod
    def from_chat_history(
        history_file: str,
        output_file: str,
        min_turns: int = 2,
    ) -> List[Dict]:
        with open(history_file, "r", encoding="utf-8") as f:
            sessions = json.load(f)

        samples = []
        for session in sessions:
            messages = session.get("messages", [])
            if len(messages) < min_turns * 2:
                continue

            for i in range(0, len(messages) - 1, 2):
                user_msg = messages[i].get("content", "") if messages[i].get("role") == "user" else ""
                ai_msg = messages[i + 1].get("content", "") if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant" else ""
                if user_msg and ai_msg and len(ai_msg) > 20:
                    samples.append({
                        "instruction": "你是一个运维专家，请根据知识回答运维问题。",
                        "input": user_msg,
                        "output": ai_msg,
                        "source": "chat_history",
                    })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        logger.info(f"从聊天记录生成 {len(samples)} 条训练数据")
        return samples

    @staticmethod
    def generate_synthetic_data(
        output_file: str,
        num_samples: int = 200,
    ) -> List[Dict]:
        templates = [
            {
                "input": "Redis 内存溢出如何排查？",
                "output": "【故障现象】Redis 进程占用内存持续增长，触发 OOM 或达到 maxmemory 上限。\n【可能原因】1. 未设置 maxmemory 或淘汰策略 2. 大 Key 堆积 3. 客户端连接泄漏 4. 持久化 fork 导致 COW 内存翻倍\n【排查命令】redis-cli info memory | grep used_memory_human\nredis-cli --bigkeys\nredis-cli client list | wc -l\n【修复步骤】1. 配置 maxmemory 和淘汰策略：config set maxmemory-policy allkeys-lru 2. 定期扫描大 Key 并拆分 3. 设置客户端超时：config set timeout 300 4. 优化持久化配置，启用 lazyfree\n【验证方法】redis-cli info memory 确认 used_memory_rss_human 恢复正常",
            },
            {
                "input": "MySQL 主从同步延迟怎么处理？",
                "output": "【故障现象】从库 Seconds_Behind_Master 持续增大，数据不一致。\n【可能原因】1. 主库大事务 2. 从库硬件性能不足 3. 网络延迟 4. 单线程复制瓶颈\n【排查命令】show slave status\\G  -- 查看 Seconds_Behind_Master\nshow processlist  -- 查看从库线程状态\n【修复步骤】1. 开启多线程复制：set global slave_parallel_workers=4 2. 拆分大事务为小批次 3. 升级从库硬件 4. 使用半同步复制保证一致性\n【验证方法】show slave status\\G 确认 Seconds_Behind_Master=0",
            },
            {
                "input": "Nginx 502 Bad Gateway 如何排查？",
                "output": "【故障现象】用户访问返回 502 Bad Gateway 错误。\n【可能原因】1. 后端服务未启动或崩溃 2. 后端服务超时 3. Nginx upstream 配置错误 4. 后端服务端口变更\n【排查命令】curl -I http://backend:8080/health\ntail -f /var/log/nginx/error.log\nnetstat -tlnp | grep 8080\n【修复步骤】1. 重启后端服务：systemctl restart app 2. 调整超时：proxy_read_timeout 60s 3. 检查 upstream 配置是否指向正确端口 4. 配置健康检查：max_fails=3 fail_timeout=30s\n【验证方法】curl -I 确认返回 200",
            },
            {
                "input": "Kubernetes Pod CrashLoopBackOff 如何解决？",
                "output": "【故障现象】Pod 反复重启，状态为 CrashLoopBackOff。\n【可能原因】1. 容器启动命令错误 2. OOMKilled 3. 配置缺失导致应用启动失败 4. 健康检查配置不当\n【排查命令】kubectl describe pod <pod-name>\nkubectl logs <pod-name> --previous\nkubectl get events --sort-by=.metadata.creationTimestamp\n【修复步骤】1. 查看上次崩溃日志定位原因 2. 调整资源限制：resources.limits.memory 3. 修复配置缺失或环境变量 4. 调整 initialDelaySeconds 延迟健康检查\n【验证方法】kubectl get pod 确认 Running 状态持续 5 分钟以上",
            },
            {
                "input": "Linux 磁盘空间满如何清理？",
                "output": "【故障现象】磁盘使用率 100%，服务无法写入文件。\n【可能原因】1. 日志文件未轮转 2. 临时文件堆积 3. 已删除文件被进程占用 4. Docker 镜像/容器堆积\n【排查命令】df -h\ndu -sh /* | sort -rh | head -10\nlsof | grep deleted\ndocker system df\n【修复步骤】1. 清理日志：find /var/log -name '*.log.*' -mtime +7 -delete 2. 清理临时文件：rm -rf /tmp/* 3. 释放已删除文件：kill 占用进程 4. Docker 清理：docker system prune -af\n【验证方法】df -h 确认使用率低于 80%",
            },
            {
                "input": "Docker 容器网络不通怎么排查？",
                "output": "【故障现象】容器间无法通信或无法访问外部网络。\n【可能原因】1. Docker 网桥配置错误 2. iptables 规则冲突 3. 容器不在同一网络 4. DNS 解析失败\n【排查命令】docker network ls\ndocker network inspect <network-name>\ndocker exec <container> ping <target>\ndocker exec <container> nslookup <domain>\n【修复步骤】1. 将容器加入同一网络：docker network connect <net> <container> 2. 重建 Docker 网络：docker network create --driver bridge <net> 3. 重启 Docker 服务恢复 iptables 4. 指定 DNS：docker run --dns 8.8.8.8\n【验证方法】docker exec <container> curl -I http://target:port",
            },
            {
                "input": "Elasticsearch 集群变红怎么处理？",
                "output": "【故障现象】Elasticsearch 集群状态为 red，部分主分片不可用。\n【可能原因】1. 节点宕机 2. 磁盘满导致分片分配失败 3. 分片损坏 4. 集群配置不当\n【排查命令】curl localhost:9200/_cluster/health?pretty\ncurl localhost:9200/_cat/shards?v&h=index,shard,state,node | grep UNASSIGNED\ncurl localhost:9200/_cat/allocation?v\n【修复步骤】1. 重启宕机节点 2. 清理磁盘空间或调整 cluster.routing.allocation.disk.threshold 3. 手动分配分片：_cluster/reroute 4. 增加副本数保证高可用\n【验证方法】curl localhost:9200/_cluster/health 确认 status=green",
            },
            {
                "input": "CPU 使用率突然飙高怎么排查？",
                "output": "【故障现象】服务器 CPU 使用率突然飙升至 90% 以上。\n【可能原因】1. 进程死循环 2. 突发流量 3. 定时任务执行 4. 内存不足导致频繁 swap\n【排查命令】top -c -o %CPU\nps aux --sort=-%cpu | head -20\nvmstat 1 5\niostat -x 1 3\n【修复步骤】1. 定位高 CPU 进程：top -c -p <PID> 2. 分析线程：top -H -p <PID> 3. 限流或扩容应对突发流量 4. 调整定时任务到低峰期 5. 优化代码或增加资源\n【验证方法】top 确认 CPU 使用率恢复到正常水平",
            },
            {
                "input": "Zookeeper 连接超时怎么处理？",
                "output": "【故障现象】客户端连接 Zookeeper 超时，服务注册发现失败。\n【可能原因】1. Zookeeper 节点负载过高 2. 网络延迟 3. session timeout 配置过小 4. JVM GC 停顿\n【排查命令】echo ruok | nc zk-host 2181\necho mntr | nc zk-host 2181\nzkCli.sh -server zk-host:2181\n【修复步骤】1. 增大 session timeout：zkSessionTimeout=30000 2. 优化 JVM 堆内存和 GC 策略 3. 检查网络延迟和带宽 4. 扩容 Zookeeper 集群\n【验证方法】客户端连接成功且无超时日志",
            },
            {
                "input": "Kafka 消费积压如何处理？",
                "output": "【故障现象】Kafka 消费者 Lag 持续增大，消息处理延迟。\n【可能原因】1. 消费者处理速度慢 2. 消费者实例不足 3. 消息量突增 4. 消费者频繁 Rebalance\n【排查命令】kafka-consumer-groups.sh --describe --group <group-id> --bootstrap-server localhost:9092\nkafka-topics.sh --describe --topic <topic>\n【修复步骤】1. 增加消费者实例数 2. 优化消费逻辑减少处理耗时 3. 临时扩容分区数 4. 调整 max.poll.interval.ms 避免 Rebalance 5. 紧急情况跳过积压：seekToBeginning 或 seekToEnd\n【验证方法】kafka-consumer-groups.sh 确认 Lag 趋近于 0",
            },
        ]

        samples = []
        for i in range(num_samples):
            t = templates[i % len(templates)]
            samples.append({
                "instruction": "你是一个运维专家，请根据知识回答运维问题。",
                "input": t["input"],
                "output": t["output"],
                "source": "synthetic",
            })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        logger.info(f"生成 {len(samples)} 条合成训练数据")
        return samples

    @staticmethod
    def _extract_query_from_chunk(content: str) -> str:
        lines = content.strip().split("\n")
        first_line = lines[0].strip()
        if "案例" in first_line or "故障" in first_line:
            return first_line.replace("案例", "").replace("：", "").replace(":", "").strip()
        keywords = content[:60].replace("\n", " ").strip()
        return f"关于{keywords}的运维问题"

    @staticmethod
    def _format_ops_answer(content: str) -> str:
        return content.strip()


class LoRATrainer:
    """LoRA 微调训练器：支持 Qwen2 系列模型的运维领域适配"""

    def __init__(
        self,
        model_path: str,
        output_dir: str = "./lora-ops-output",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        use_gpu: bool = True,
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.use_gpu = use_gpu and _check_gpu_available()

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        logger.info(f"LoRA 配置: r={lora_r}, alpha={lora_alpha}, targets={self.lora_config.target_modules}")
        logger.info(f"设备: {'GPU' if self.use_gpu else 'CPU'}")

    def prepare_dataset(self, data_file: str, test_ratio: float = 0.1, max_length: int = 512):
        from datasets import Dataset as HFDataset
        from sklearn.model_selection import train_test_split

        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        logger.info(f"加载 {len(raw_data)} 条原始数据")

        tokenizer = self._load_tokenizer()

        def format_chat(sample):
            return (
                f"<|im_start|>system\n{sample['instruction']}<|im_end|>\n"
                f"<|im_start|>user\n{sample['input']}<|im_end|>\n"
                f"<|im_start|>assistant\n{sample['output']}<|im_end|>"
            )

        texts = [format_chat(s) for s in raw_data]

        train_texts, val_texts = train_test_split(texts, test_size=test_ratio, random_state=42)
        logger.info(f"训练集: {len(train_texts)}, 验证集: {len(val_texts)}")

        def tokenize_fn(examples):
            outputs = tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            outputs["labels"] = [
                [(tid if tid != tokenizer.pad_token_id else -100) for tid in ids]
                for ids in outputs["input_ids"]
            ]
            return outputs

        train_ds = HFDataset.from_dict({"text": train_texts})
        val_ds = HFDataset.from_dict({"text": val_texts})

        train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

        return train_ds, val_ds, tokenizer

    def train(
        self,
        train_dataset,
        val_dataset,
        tokenizer,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 4,
    ):
        import torch
        from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

        logger.info(f"加载基座模型: {self.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self.use_gpu else torch.float32,
            device_map="auto" if self.use_gpu else None,
            trust_remote_code=True,
        )

        if not self.use_gpu:
            model = model.to("cpu")

        from peft import get_peft_model
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=self.use_gpu,
            bf16=False,
            report_to="none",
            dataloader_pin_memory=self.use_gpu,
            gradient_checkpointing=True,
            optim="adamw_torch",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        )

        logger.info("开始训练...")
        train_result = trainer.train()

        metrics = train_result.metrics
        logger.info(f"训练完成: {metrics}")

        trainer.save_model(os.path.join(self.output_dir, "best_model"))
        tokenizer.save_pretrained(os.path.join(self.output_dir, "best_model"))

        eval_metrics = trainer.evaluate()
        logger.info(f"验证集评估: {eval_metrics}")

        return model, eval_metrics

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer


class LoRAInference:
    """LoRA 微调模型推理器：加载基座+LoRA权重进行推理"""

    def __init__(self, base_model_path: str, lora_path: str, use_gpu: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.device = "cuda" if (use_gpu and _check_gpu_available()) else "cpu"

        logger.info(f"加载基座模型: {base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        logger.info(f"加载 LoRA 权重: {lora_path}")
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model.eval()

        if self.device == "cpu":
            self.model = self.model.to("cpu")

        logger.info("模型加载完成")

    def generate(self, query: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        import torch

        prompt = (
            f"<|im_start|>system\n你是一个运维专家，请根据知识回答运维问题。<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()

    def batch_generate(self, queries: List[str], **kwargs) -> List[str]:
        return [self.generate(q, **kwargs) for q in queries]


class RerankerFinetuner:
    """Reranker 微调器：针对运维领域优化重排序模型"""

    def __init__(
        self,
        base_model_path: str,
        output_dir: str = "./lora-reranker-output",
        use_gpu: bool = True,
    ):
        self.base_model_path = base_model_path
        self.output_dir = output_dir
        self.use_gpu = use_gpu and _check_gpu_available()

    def prepare_rerank_dataset(self, data_file: str) -> Tuple:
        from datasets import Dataset as HFDataset
        from sentence_transformers import InputExample

        with open(data_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        samples = []
        for item in raw:
            query = item["query"]
            for ctx in item.get("positive_contexts", []):
                samples.append({"query": query, "document": ctx, "label": 1.0})
            for ctx in item.get("negative_contexts", []):
                samples.append({"query": query, "document": ctx, "label": 0.0})

        logger.info(f"Reranker 训练样本: {len(samples)} (正例: {sum(1 for s in samples if s['label']==1)}, 负例: {sum(1 for s in samples if s['label']==0)})")
        return samples

    def train(self, samples, num_epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        from sentence_transformers import CrossEncoder, InputExample
        from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator

        model = CrossEncoder(self.base_model_path, num_labels=1)

        train_examples = [
            InputExample(texts=[s["query"], s["document"]], label=s["label"])
            for s in samples
        ]

        split = int(len(train_examples) * 0.9)
        train_examples_split = train_examples[:split]
        val_examples = train_examples[split:]

        model.fit(
            train_dataloader=model.smart_batching_collate_fn(train_examples_split, batch_size),
            epochs=num_epochs,
            optimizer_params={"lr": learning_rate},
            show_progress_bar=True,
        )

        model.save(os.path.join(self.output_dir, "best_reranker"))
        logger.info(f"Reranker 模型已保存至 {self.output_dir}/best_reranker")
        return model


def _check_gpu_available() -> bool:
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            logger.info(f"GPU 可用: {torch.cuda.get_device_name(0)}, 显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")
        else:
            logger.info("GPU 不可用，使用 CPU 训练")
        return available
    except ImportError:
        return False


if __name__ == "__main__":
    import argparse

    # 数据目录优先级：data/prepared/ > finetune_data/
    _prepared_dir = _BASE_DIR / "data" / "prepared"
    _finetune_dir = _BASE_DIR / "finetune_data"
    _default_lora_data = str(_prepared_dir / "lora_train.json") if (_prepared_dir / "lora_train.json").exists() else str(_finetune_dir / "ops_train.json")
    _default_rerank_data = str(_prepared_dir / "reranker_train.json") if (_prepared_dir / "reranker_train.json").exists() else str(_finetune_dir / "rerank_train.json")

    parser = argparse.ArgumentParser(description="SmartOps LoRA 微调工具")
    parser.add_argument("--mode", choices=["prepare", "train", "inference", "reranker"], default="prepare")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-1.5B-Instruct", help="基座模型路径")
    parser.add_argument("--data-file", default=_default_lora_data, help="LoRA训练数据文件")
    parser.add_argument("--rerank-data-file", default=_default_rerank_data, help="Reranker训练数据文件")
    parser.add_argument("--output-dir", default=str(_BASE_DIR / "model" / "lora-ops"))
    parser.add_argument("--pdf-path", default=str(_BASE_DIR / "data" / "文档2.pdf"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--query", type=str, default="Redis内存溢出如何排查？")
    parser.add_argument("--use-gpu", action="store_true", default=True)
    args = parser.parse_args()

    if args.mode == "prepare":
        # 优先使用 data/prepared/ 目录
        data_dir = _BASE_DIR / "data" / "prepared"
        data_dir.mkdir(parents=True, exist_ok=True)

        output = str(data_dir / "lora_train.json")

        if Path(args.pdf_path).exists():
            samples = OpsDataBuilder.from_retriever_chunks(args.pdf_path, output)
        else:
            logger.warning(f"PDF 不存在: {args.pdf_path}，使用合成数据")
            samples = OpsDataBuilder.generate_synthetic_data(output)

        logger.info(f"数据准备完成，共 {len(samples)} 条，保存至 {output}")

    elif args.mode == "train":
        trainer = LoRATrainer(
            model_path=args.model_path,
            output_dir=args.output_dir,
            lora_r=args.lora_r,
            use_gpu=args.use_gpu,
        )

        train_ds, val_ds, tokenizer = trainer.prepare_dataset(args.data_file)
        model, metrics = trainer.train(
            train_ds, val_ds, tokenizer,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

        logger.info(f"训练完成! 最佳模型保存至 {args.output_dir}/best_model")
        logger.info(f"验证指标: {metrics}")

    elif args.mode == "inference":
        lora_path = os.path.join(args.output_dir, "best_model")
        infer = LoRAInference(args.model_path, lora_path, use_gpu=args.use_gpu)
        result = infer.generate(args.query)
        print(f"\n问题: {args.query}")
        print(f"回答: {result}")

    elif args.mode == "reranker":
        reranker = RerankerFinetuner(
            base_model_path=str(_BASE_DIR / "model" / "bge-reranker-v2-m3"),
            output_dir=str(_BASE_DIR / "model" / "lora-reranker"),
        )
        # 加载新的 reranker_train.json 数据
        rerank_file = args.rerank_data_file
        if not Path(rerank_file).exists():
            logger.error(f"Reranker 训练数据不存在: {rerank_file}")
            logger.info("请先运行: python scripts/prepare_ops_data.py --task finetune")
        else:
            logger.info(f"加载 Reranker 训练数据: {rerank_file}")
            samples = reranker.prepare_rerank_dataset(rerank_file)
            model = reranker.train(samples)
            logger.info(f"Reranker 训练完成! 模型保存至 {reranker.output_dir}/best_reranker")
