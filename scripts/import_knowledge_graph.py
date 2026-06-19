"""
SmartOps 知识图谱数据导入脚本

功能：
  1. 清空 Neo4j 旧测试数据
  2. 基于运维故障知识构建实体和关系，导入 Neo4j
  3. 适配项目的实体类型（Component/Fault/Command/Config/Metric/Service/Protocol）
     和关系类型（causes/fixes/depends_on/monitors/configures/relates_to/indicates/restarts/checks）

用法：
  python scripts/import_knowledge_graph.py           # 清空旧数据 + 导入新数据
  python scripts/import_knowledge_graph.py --no-clear # 保留旧数据，追加导入
"""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

NEO4J_URI = "bolt://192.168.100.128:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "hr.sUqm9B75C.sK"

# ============================================================================
# 知识图谱三元组数据
# 格式: (from_entity, from_type, relation, to_entity, to_type)
# 实体类型: Component, Fault, Command, Config, Metric, Service, Protocol
# 关系类型: causes, fixes, depends_on, monitors, configures, relates_to, indicates, restarts, checks
# ============================================================================

TRIPLES = [
    # =========================================================================
    # Redis 故障链
    # =========================================================================
    # --- Redis 内存溢出 OOM ---
    ("Redis", "Component", "causes", "内存溢出", "Fault"),
    ("大Key堆积", "Fault", "causes", "内存溢出", "Fault"),
    ("客户端连接泄漏", "Fault", "causes", "内存溢出", "Fault"),
    ("过期Key未清理", "Fault", "causes", "内存溢出", "Fault"),
    ("maxmemory", "Config", "configures", "Redis", "Component"),
    ("maxmemory-policy", "Config", "configures", "Redis", "Component"),
    ("maxmemory", "Config", "fixes", "内存溢出", "Fault"),
    ("maxmemory-policy", "Config", "fixes", "内存溢出", "Fault"),
    ("redis-cli --bigkeys", "Command", "checks", "大Key堆积", "Fault"),
    ("redis-cli info memory", "Command", "checks", "内存溢出", "Fault"),
    ("lazyfree-lazy-eviction", "Config", "configures", "Redis", "Component"),
    ("timeout", "Config", "configures", "Redis", "Component"),
    ("timeout", "Config", "fixes", "客户端连接泄漏", "Fault"),
    ("SCAN", "Command", "fixes", "大Key堆积", "Fault"),
    ("used_memory", "Metric", "indicates", "内存溢出", "Fault"),

    # --- Redis 主从复制中断 ---
    ("Redis", "Component", "causes", "主从同步失败", "Fault"),
    ("repl-backlog-size", "Config", "configures", "Redis", "Component"),
    ("repl-backlog-size", "Config", "fixes", "主从同步失败", "Fault"),
    ("masterauth", "Config", "configures", "Redis", "Component"),
    ("masterauth", "Config", "fixes", "主从同步失败", "Fault"),
    ("repl-timeout", "Config", "configures", "Redis", "Component"),
    ("redis-cli info replication", "Command", "checks", "主从同步失败", "Fault"),
    ("slaveof", "Command", "fixes", "主从同步失败", "Fault"),
    ("master_link_status", "Metric", "indicates", "主从同步失败", "Fault"),

    # --- Redis 缓存穿透/击穿/雪崩 ---
    ("缓存穿透", "Fault", "causes", "数据库压力过大", "Fault"),
    ("缓存击穿", "Fault", "causes", "数据库压力过大", "Fault"),
    ("缓存雪崩", "Fault", "causes", "数据库压力过大", "Fault"),
    ("布隆过滤器", "Service", "fixes", "缓存穿透", "Fault"),
    ("缓存空值", "Config", "fixes", "缓存穿透", "Fault"),
    ("分布式锁", "Service", "fixes", "缓存击穿", "Fault"),
    ("setnx", "Command", "fixes", "缓存击穿", "Fault"),
    ("多级缓存", "Service", "fixes", "缓存雪崩", "Fault"),
    ("Redis Sentinel", "Service", "fixes", "缓存雪崩", "Fault"),
    ("Redis Cluster", "Service", "fixes", "缓存雪崩", "Fault"),
    ("缓存命中率", "Metric", "indicates", "缓存穿透", "Fault"),
    ("缓存命中率", "Metric", "indicates", "缓存击穿", "Fault"),
    ("缓存命中率", "Metric", "indicates", "缓存雪崩", "Fault"),

    # --- Redis 持久化 ---
    ("Redis", "Component", "causes", "RDB保存失败", "Fault"),
    ("Redis", "Component", "causes", "AOF重写失败", "Fault"),
    ("save", "Config", "configures", "Redis", "Component"),
    ("appendonly", "Config", "configures", "Redis", "Component"),
    ("appendfsync", "Config", "configures", "Redis", "Component"),
    ("BGSAVE", "Command", "fixes", "RDB保存失败", "Fault"),
    ("BGREWRITEAOF", "Command", "fixes", "AOF重写失败", "Fault"),
    ("redis-check-aof", "Command", "fixes", "AOF重写失败", "Fault"),
    ("redis-check-rdb", "Command", "fixes", "RDB保存失败", "Fault"),

    # --- Redis Cluster ---
    ("Redis Cluster", "Service", "causes", "脑裂", "Fault"),
    ("Redis Cluster", "Service", "causes", "集群节点故障", "Fault"),
    ("redis-cli --cluster reshard", "Command", "fixes", "集群节点故障", "Fault"),
    ("redis-cli --cluster add-node", "Command", "fixes", "集群节点故障", "Fault"),
    ("cluster_state", "Metric", "indicates", "集群节点故障", "Fault"),

    # =========================================================================
    # MySQL 故障链
    # =========================================================================
    # --- MySQL 主从同步延迟 ---
    ("MySQL", "Component", "causes", "主从同步延迟", "Fault"),
    ("大事务", "Fault", "causes", "主从同步延迟", "Fault"),
    ("从库性能不足", "Fault", "causes", "主从同步延迟", "Fault"),
    ("单线程回放", "Fault", "causes", "主从同步延迟", "Fault"),
    ("slave_parallel_workers", "Config", "configures", "MySQL", "Component"),
    ("slave_parallel_workers", "Config", "fixes", "主从同步延迟", "Fault"),
    ("slave_parallel_type", "Config", "configures", "MySQL", "Component"),
    ("SHOW SLAVE STATUS", "Command", "checks", "主从同步延迟", "Fault"),
    ("Seconds_Behind_Master", "Metric", "indicates", "主从同步延迟", "Fault"),
    ("innodb_buffer_pool_size", "Config", "configures", "MySQL", "Component"),
    ("innodb_buffer_pool_size", "Config", "fixes", "从库性能不足", "Fault"),

    # --- MySQL 慢查询 ---
    ("MySQL", "Component", "causes", "慢查询", "Fault"),
    ("缺少索引", "Fault", "causes", "慢查询", "Fault"),
    ("SQL写法不当", "Fault", "causes", "慢查询", "Fault"),
    ("表数据量过大", "Fault", "causes", "慢查询", "Fault"),
    ("slow_query_log", "Config", "configures", "MySQL", "Component"),
    ("long_query_time", "Config", "configures", "MySQL", "Component"),
    ("EXPLAIN", "Command", "checks", "慢查询", "Fault"),
    ("mysqldumpslow", "Command", "checks", "慢查询", "Fault"),
    ("ALTER TABLE ADD INDEX", "Command", "fixes", "缺少索引", "Fault"),
    ("慢查询数量", "Metric", "indicates", "慢查询", "Fault"),

    # --- MySQL 死锁 ---
    ("MySQL", "Component", "causes", "死锁", "Fault"),
    ("资源访问顺序不一致", "Fault", "causes", "死锁", "Fault"),
    ("外键约束", "Config", "causes", "死锁", "Fault"),
    ("SHOW ENGINE INNODB STATUS", "Command", "checks", "死锁", "Fault"),
    ("innodb_print_all_deadlocks", "Config", "configures", "MySQL", "Component"),
    ("统一访问顺序", "Command", "fixes", "死锁", "Fault"),

    # --- MySQL 磁盘满 ---
    ("MySQL", "Component", "causes", "磁盘满", "Fault"),
    ("二进制日志堆积", "Fault", "causes", "磁盘满", "Fault"),
    ("ibdata1文件过大", "Fault", "causes", "磁盘满", "Fault"),
    ("PURGE BINARY LOGS", "Command", "fixes", "二进制日志堆积", "Fault"),
    ("OPTIMIZE TABLE", "Command", "fixes", "ibdata1文件过大", "Fault"),
    ("innodb_file_per_table", "Config", "configures", "MySQL", "Component"),
    ("innodb_file_per_table", "Config", "fixes", "ibdata1文件过大", "Fault"),
    ("磁盘使用率", "Metric", "indicates", "磁盘满", "Fault"),

    # --- MySQL 连接数过多 ---
    ("MySQL", "Component", "causes", "连接数过多", "Fault"),
    ("max_connections", "Config", "configures", "MySQL", "Component"),
    ("max_connections", "Config", "fixes", "连接数过多", "Fault"),
    ("wait_timeout", "Config", "configures", "MySQL", "Component"),
    ("wait_timeout", "Config", "fixes", "连接数过多", "Fault"),
    ("SHOW PROCESSLIST", "Command", "checks", "连接数过多", "Fault"),
    ("thread_cache_size", "Config", "configures", "MySQL", "Component"),
    ("连接数", "Metric", "indicates", "连接数过多", "Fault"),

    # =========================================================================
    # Nginx 故障链
    # =========================================================================
    ("Nginx", "Component", "causes", "502错误", "Fault"),
    ("后端服务不可用", "Fault", "causes", "502错误", "Fault"),
    ("后端服务超时", "Fault", "causes", "502错误", "Fault"),
    ("upstream配置错误", "Fault", "causes", "502错误", "Fault"),
    ("worker_connections", "Config", "configures", "Nginx", "Component"),
    ("proxy_connect_timeout", "Config", "configures", "Nginx", "Component"),
    ("proxy_read_timeout", "Config", "configures", "Nginx", "Component"),
    ("proxy_connect_timeout", "Config", "fixes", "后端服务超时", "Fault"),
    ("nginx -t", "Command", "checks", "upstream配置错误", "Fault"),
    ("systemctl restart", "Command", "restarts", "Nginx", "Component"),
    ("keepalive_timeout", "Config", "configures", "Nginx", "Component"),
    ("Nginx", "Component", "causes", "504超时", "Fault"),
    ("后端服务超时", "Fault", "causes", "504超时", "Fault"),
    ("proxy_read_timeout", "Config", "fixes", "504超时", "Fault"),

    # =========================================================================
    # Kubernetes 故障链
    # =========================================================================
    # --- Pod CrashLoopBackOff ---
    ("Kubernetes", "Component", "causes", "CrashLoopBackOff", "Fault"),
    ("容器启动命令错误", "Fault", "causes", "CrashLoopBackOff", "Fault"),
    ("OOMKilled", "Fault", "causes", "CrashLoopBackOff", "Fault"),
    ("Liveness Probe配置不当", "Fault", "causes", "CrashLoopBackOff", "Fault"),
    ("配置缺失", "Fault", "causes", "CrashLoopBackOff", "Fault"),
    ("resources.limits.memory", "Config", "configures", "Kubernetes", "Component"),
    ("resources.limits.memory", "Config", "fixes", "OOMKilled", "Fault"),
    ("kubectl describe pod", "Command", "checks", "CrashLoopBackOff", "Fault"),
    ("kubectl logs --previous", "Command", "checks", "CrashLoopBackOff", "Fault"),
    ("initialDelaySeconds", "Config", "configures", "Kubernetes", "Component"),
    ("initialDelaySeconds", "Config", "fixes", "Liveness Probe配置不当", "Fault"),
    ("RestartCount", "Metric", "indicates", "CrashLoopBackOff", "Fault"),

    # --- Node NotReady ---
    ("Kubernetes", "Component", "causes", "NotReady", "Fault"),
    ("kubelet服务停止", "Fault", "causes", "NotReady", "Fault"),
    ("磁盘压力", "Fault", "causes", "NotReady", "Fault"),
    ("内存压力", "Fault", "causes", "NotReady", "Fault"),
    ("CNI网络插件故障", "Fault", "causes", "NotReady", "Fault"),
    ("systemctl restart kubelet", "Command", "fixes", "kubelet服务停止", "Fault"),
    ("kubectl describe node", "Command", "checks", "NotReady", "Fault"),
    ("NodeCondition", "Metric", "indicates", "NotReady", "Fault"),

    # --- Service/Ingress ---
    ("Kubernetes", "Component", "causes", "Service无法访问", "Fault"),
    ("selector不匹配", "Fault", "causes", "Service无法访问", "Fault"),
    ("端口配置错误", "Fault", "causes", "Service无法访问", "Fault"),
    ("Endpoints为空", "Fault", "causes", "Service无法访问", "Fault"),
    ("kubectl get endpoints", "Command", "checks", "Service无法访问", "Fault"),
    ("kubectl describe svc", "Command", "checks", "Service无法访问", "Fault"),

    # --- K8s 资源配额 ---
    ("Kubernetes", "Component", "causes", "Pod无法调度", "Fault"),
    ("资源不足", "Fault", "causes", "Pod无法调度", "Fault"),
    ("ResourceQuota超限", "Fault", "causes", "Pod无法调度", "Fault"),
    ("ResourceQuota", "Config", "configures", "Kubernetes", "Component"),
    ("LimitRange", "Config", "configures", "Kubernetes", "Component"),
    ("kubectl describe resourcequota", "Command", "checks", "ResourceQuota超限", "Fault"),
    ("kubectl top nodes", "Command", "checks", "资源不足", "Fault"),

    # =========================================================================
    # Docker 故障链
    # =========================================================================
    ("Docker", "Component", "causes", "容器网络不通", "Fault"),
    ("Docker网桥配置错误", "Fault", "causes", "容器网络不通", "Fault"),
    ("iptables规则冲突", "Fault", "causes", "容器网络不通", "Fault"),
    ("DNS解析失败", "Fault", "causes", "容器网络不通", "Fault"),
    ("docker network connect", "Command", "fixes", "容器网络不通", "Fault"),
    ("docker network inspect", "Command", "checks", "容器网络不通", "Fault"),
    ("--dns", "Config", "configures", "Docker", "Component"),
    ("--dns", "Config", "fixes", "DNS解析失败", "Fault"),
    ("--memory", "Config", "configures", "Docker", "Component"),
    ("--cpus", "Config", "configures", "Docker", "Component"),
    ("Docker", "Component", "causes", "磁盘空间不足", "Fault"),
    ("Docker镜像堆积", "Fault", "causes", "磁盘空间不足", "Fault"),
    ("docker system prune", "Command", "fixes", "Docker镜像堆积", "Fault"),
    ("docker system df", "Command", "checks", "磁盘空间不足", "Fault"),

    # =========================================================================
    # Linux 故障链
    # =========================================================================
    ("Linux", "Component", "causes", "磁盘空间满", "Fault"),
    ("日志文件未轮转", "Fault", "causes", "磁盘空间满", "Fault"),
    ("已删除文件被进程占用", "Fault", "causes", "磁盘空间满", "Fault"),
    ("inode耗尽", "Fault", "causes", "磁盘空间满", "Fault"),
    ("df -h", "Command", "checks", "磁盘空间满", "Fault"),
    ("du -sh", "Command", "checks", "磁盘空间满", "Fault"),
    ("lsof", "Command", "checks", "已删除文件被进程占用", "Fault"),
    ("logrotate", "Command", "fixes", "日志文件未轮转", "Fault"),

    ("Linux", "Component", "causes", "CPU过高", "Fault"),
    ("进程死循环", "Fault", "causes", "CPU过高", "Fault"),
    ("突发流量", "Fault", "causes", "CPU过高", "Fault"),
    ("top", "Command", "checks", "CPU过高", "Fault"),
    ("vm.overcommit_memory", "Config", "configures", "Linux", "Component"),
    ("vm.swappiness", "Config", "configures", "Linux", "Component"),
    ("net.core.somaxconn", "Config", "configures", "Linux", "Component"),
    ("fs.file-max", "Config", "configures", "Linux", "Component"),

    ("Linux", "Component", "causes", "内存泄漏", "Fault"),
    ("进程内存持续增长", "Fault", "causes", "内存泄漏", "Fault"),
    ("OOM Killer", "Fault", "causes", "进程被杀", "Fault"),
    ("free -m", "Command", "checks", "内存泄漏", "Fault"),
    ("pmap", "Command", "checks", "内存泄漏", "Fault"),
    ("valgrind", "Command", "fixes", "内存泄漏", "Fault"),
    ("oom_score_adj", "Config", "configures", "Linux", "Component"),

    ("Linux", "Component", "causes", "网络超时", "Fault"),
    ("DNS解析慢", "Fault", "causes", "网络超时", "Fault"),
    ("防火墙阻拦", "Fault", "causes", "网络超时", "Fault"),
    ("TCP连接队列满", "Fault", "causes", "网络超时", "Fault"),
    ("TIME_WAIT过多", "Fault", "causes", "网络超时", "Fault"),
    ("nslookup", "Command", "checks", "DNS解析慢", "Fault"),
    ("iptables -L", "Command", "checks", "防火墙阻拦", "Fault"),
    ("ss -s", "Command", "checks", "TIME_WAIT过多", "Fault"),
    ("tcp_tw_reuse", "Config", "fixes", "TIME_WAIT过多", "Fault"),

    # =========================================================================
    # Kafka 故障链
    # =========================================================================
    ("Kafka", "Component", "causes", "消费积压", "Fault"),
    ("消费者处理速度慢", "Fault", "causes", "消费积压", "Fault"),
    ("消费者实例不足", "Fault", "causes", "消费积压", "Fault"),
    ("消息量突增", "Fault", "causes", "消费积压", "Fault"),
    ("频繁Rebalance", "Fault", "causes", "消费积压", "Fault"),
    ("max.poll.interval.ms", "Config", "configures", "Kafka", "Component"),
    ("max.poll.interval.ms", "Config", "fixes", "频繁Rebalance", "Fault"),
    ("kafka-consumer-groups.sh", "Command", "checks", "消费积压", "Fault"),
    ("ConsumerLag", "Metric", "indicates", "消费积压", "Fault"),

    ("Kafka", "Component", "causes", "消息丢失", "Fault"),
    ("acks未设为all", "Fault", "causes", "消息丢失", "Fault"),
    ("acks", "Config", "configures", "Kafka", "Component"),
    ("acks", "Config", "fixes", "消息丢失", "Fault"),
    ("min.insync.replicas", "Config", "configures", "Kafka", "Component"),
    ("replication.factor", "Config", "configures", "Kafka", "Component"),

    # =========================================================================
    # Elasticsearch 故障链
    # =========================================================================
    ("Elasticsearch", "Component", "causes", "集群变红", "Fault"),
    ("节点宕机", "Fault", "causes", "集群变红", "Fault"),
    ("磁盘满", "Fault", "causes", "集群变红", "Fault"),
    ("分片损坏", "Fault", "causes", "集群变红", "Fault"),
    ("_cluster/health", "Command", "checks", "集群变红", "Fault"),
    ("_cat/shards", "Command", "checks", "分片损坏", "Fault"),
    ("_cluster/reroute", "Command", "fixes", "分片损坏", "Fault"),
    ("cluster.routing.allocation.disk.threshold", "Config", "configures", "Elasticsearch", "Component"),
    ("cluster_status", "Metric", "indicates", "集群变红", "Fault"),

    ("Elasticsearch", "Component", "causes", "慢查询", "Fault"),
    ("索引设计不当", "Fault", "causes", "慢查询", "Fault"),
    ("mapping不合理", "Fault", "causes", "慢查询", "Fault"),
    ("分片数过多", "Fault", "causes", "慢查询", "Fault"),
    ("_search?explain", "Command", "checks", "慢查询", "Fault"),
    ("refresh_interval", "Config", "configures", "Elasticsearch", "Component"),
    ("number_of_shards", "Config", "configures", "Elasticsearch", "Component"),

    # =========================================================================
    # RabbitMQ 故障链
    # =========================================================================
    ("RabbitMQ", "Component", "causes", "队列积压", "Fault"),
    ("消费者宕机", "Fault", "causes", "队列积压", "Fault"),
    ("消息处理异常", "Fault", "causes", "队列积压", "Fault"),
    ("rabbitmqctl list_queues", "Command", "checks", "队列积压", "Fault"),
    ("publisher confirm", "Protocol", "fixes", "消息丢失", "Fault"),
    ("Quorum Queue", "Service", "fixes", "脑裂", "Fault"),
    ("RabbitMQ", "Component", "causes", "脑裂", "Fault"),

    # =========================================================================
    # MongoDB 故障链
    # =========================================================================
    ("MongoDB", "Component", "causes", "副本集选举失败", "Fault"),
    ("节点间网络分区", "Fault", "causes", "副本集选举失败", "Fault"),
    ("节点磁盘满", "Fault", "causes", "副本集选举失败", "Fault"),
    ("rs.status()", "Command", "checks", "副本集选举失败", "Fault"),
    ("rs.reconfig()", "Command", "fixes", "副本集选举失败", "Fault"),

    # =========================================================================
    # Prometheus 监控链
    # =========================================================================
    ("Prometheus", "Component", "monitors", "Redis", "Component"),
    ("Prometheus", "Component", "monitors", "MySQL", "Component"),
    ("Prometheus", "Component", "monitors", "Nginx", "Component"),
    ("Prometheus", "Component", "monitors", "Kubernetes", "Component"),
    ("Prometheus", "Component", "monitors", "Linux", "Component"),
    ("Prometheus", "Component", "monitors", "Kafka", "Component"),
    ("Prometheus", "Component", "monitors", "Elasticsearch", "Component"),
    ("Prometheus", "Component", "monitors", "Docker", "Component"),
    ("Grafana", "Component", "depends_on", "Prometheus", "Component"),
    ("AlertManager", "Service", "depends_on", "Prometheus", "Component"),
    ("Prometheus", "Component", "causes", "告警风暴", "Fault"),
    ("告警规则过多", "Fault", "causes", "告警风暴", "Fault"),
    ("分组抑制配置不当", "Fault", "causes", "告警风暴", "Fault"),
    ("group_by", "Config", "configures", "AlertManager", "Service"),
    ("inhibit_rules", "Config", "configures", "AlertManager", "Service"),

    # =========================================================================
    # 组件间依赖关系
    # =========================================================================
    ("Nginx", "Component", "depends_on", "后端服务", "Service"),
    ("Kafka", "Component", "depends_on", "Zookeeper", "Component"),
    ("Kubernetes", "Component", "depends_on", "etcd", "Component"),
    ("Redis Cluster", "Service", "depends_on", "Redis", "Component"),
    ("MySQL MHA", "Service", "depends_on", "MySQL", "Component"),
    ("Elasticsearch", "Component", "depends_on", "Java", "Component"),
    ("Kafka", "Component", "depends_on", "Java", "Component"),
    ("Logstash", "Service", "depends_on", "Elasticsearch", "Component"),
    ("Kibana", "Service", "depends_on", "Elasticsearch", "Component"),

    # =========================================================================
    # Java 应用故障链
    # =========================================================================
    ("Java", "Component", "causes", "OOM", "Fault"),
    ("Java", "Component", "causes", "CPU过高", "Fault"),
    ("内存泄漏", "Fault", "causes", "OOM", "Fault"),
    ("堆内存不足", "Fault", "causes", "OOM", "Fault"),
    ("jstack", "Command", "checks", "CPU过高", "Fault"),
    ("jstat", "Command", "checks", "OOM", "Fault"),
    ("Arthas", "Service", "checks", "CPU过高", "Fault"),
    ("-Xmx", "Config", "configures", "Java", "Component"),
    ("-Xms", "Config", "configures", "Java", "Component"),
    ("-XX:+UseG1GC", "Config", "configures", "Java", "Component"),
    ("-XX:+UseZGC", "Config", "configures", "Java", "Component"),
    ("GC频率", "Metric", "indicates", "OOM", "Fault"),

    # =========================================================================
    # 网络故障链
    # =========================================================================
    ("TCP", "Protocol", "causes", "连接超时", "Fault"),
    ("TCP", "Protocol", "causes", "TIME_WAIT过多", "Fault"),
    ("DNS", "Protocol", "causes", "DNS解析慢", "Fault"),
    ("SSL/TLS", "Protocol", "causes", "证书过期", "Fault"),
    ("tcpdump", "Command", "checks", "连接超时", "Fault"),
    ("traceroute", "Command", "checks", "网络超时", "Fault"),
    ("openssl s_client", "Command", "checks", "证书过期", "Fault"),
    ("tcp_tw_reuse", "Config", "configures", "Linux", "Component"),
    ("tcp_max_syn_backlog", "Config", "configures", "Linux", "Component"),

    # =========================================================================
    # 微服务故障链
    # =========================================================================
    ("微服务", "Service", "causes", "调用链路超时", "Fault"),
    ("下游服务慢", "Fault", "causes", "调用链路超时", "Fault"),
    ("服务熔断", "Service", "fixes", "调用链路超时", "Fault"),
    ("服务降级", "Service", "fixes", "调用链路超时", "Fault"),
    ("Jaeger", "Service", "checks", "调用链路超时", "Fault"),
    ("Sentinel", "Service", "fixes", "调用链路超时", "Fault"),
    ("Hystrix", "Service", "fixes", "调用链路超时", "Fault"),

    # =========================================================================
    # 部署相关
    # =========================================================================
    ("蓝绿部署", "Service", "relates_to", "Nginx", "Component"),
    ("金丝雀发布", "Service", "relates_to", "Kubernetes", "Component"),
    ("滚动更新", "Service", "relates_to", "Kubernetes", "Component"),
    ("kubectl apply", "Command", "fixes", "Pod无法调度", "Fault"),
    ("kubectl rollout restart", "Command", "restarts", "Kubernetes", "Component"),

    # =========================================================================
    # Consul / Nacos 服务发现
    # =========================================================================
    ("Consul", "Component", "monitors", "微服务", "Service"),
    ("Nacos", "Component", "monitors", "微服务", "Service"),
    ("Consul", "Component", "relates_to", "Prometheus", "Component"),
    ("Nacos", "Component", "relates_to", "Kubernetes", "Component"),

    # =========================================================================
    # 通用运维指标
    # =========================================================================
    ("CPU使用率", "Metric", "indicates", "CPU过高", "Fault"),
    ("内存使用率", "Metric", "indicates", "内存溢出", "Fault"),
    ("磁盘使用率", "Metric", "indicates", "磁盘满", "Fault"),
    ("网络延迟", "Metric", "indicates", "网络超时", "Fault"),
    ("错误率", "Metric", "indicates", "502错误", "Fault"),
    ("QPS", "Metric", "indicates", "CPU过高", "Fault"),
    ("响应时间", "Metric", "indicates", "慢查询", "Fault"),
    ("连接数", "Metric", "indicates", "连接数过多", "Fault"),
]


def clear_neo4j(driver):
    """清空 Neo4j 所有节点和关系"""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        logger.info("[Neo4j] 已清空所有节点和关系")


def create_constraints(driver):
    """创建唯一性约束"""
    entity_types = ["Component", "Fault", "Command", "Config", "Metric", "Service", "Protocol"]
    with driver.session() as session:
        for et in entity_types:
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{et}) REQUIRE n.name IS UNIQUE"
            )
    logger.info(f"[Neo4j] 已创建 {len(entity_types)} 个唯一性约束")


def import_triples(driver, triples):
    """批量导入三元组"""
    # 先收集所有实体和关系
    entities = {}  # (name, type) -> properties
    relations = []  # (from_name, from_type, relation, to_name, to_type)

    for from_name, from_type, relation, to_name, to_type in triples:
        entities[(from_name, from_type)] = {"name": from_name}
        entities[(to_name, to_type)] = {"name": to_name}
        relations.append((from_name, from_type, relation, to_name, to_type))

    # 批量创建实体
    entity_count = 0
    with driver.session() as session:
        for (name, etype), props in entities.items():
            try:
                session.run(
                    f"MERGE (e:{etype} {{name: $name}}) SET e += $props",
                    name=name,
                    props=props,
                )
                entity_count += 1
            except Exception as e:
                logger.warning(f"[Neo4j] 创建实体失败 {etype}:{name} - {e}")

    logger.info(f"[Neo4j] 已创建 {entity_count} 个实体")

    # 批量创建关系
    relation_count = 0
    with driver.session() as session:
        for from_name, from_type, relation, to_name, to_type in relations:
            try:
                # 使用参数化查询避免 Cypher 注入
                query = (
                    f"MATCH (a:{from_type} {{name: $from_name}}) "
                    f"MATCH (b:{to_type} {{name: $to_name}}) "
                    f"MERGE (a)-[r:{relation}]->(b)"
                )
                session.run(query, from_name=from_name, to_name=to_name)
                relation_count += 1
            except Exception as e:
                logger.warning(f"[Neo4j] 创建关系失败 {from_type}:{from_name}-[{relation}]->{to_type}:{to_name} - {e}")

    logger.info(f"[Neo4j] 已创建 {relation_count} 个关系")
    return entity_count, relation_count


def verify_graph(driver):
    """验证知识图谱数据"""
    with driver.session() as session:
        # 统计各类型实体数量
        entity_types = ["Component", "Fault", "Command", "Config", "Metric", "Service", "Protocol"]
        logger.info("=" * 60)
        logger.info("知识图谱数据验证")
        logger.info("=" * 60)

        total_entities = 0
        for et in entity_types:
            result = session.run(f"MATCH (n:{et}) RETURN count(n) AS cnt")
            cnt = result.single()["cnt"]
            total_entities += cnt
            if cnt > 0:
                logger.info(f"  {et}: {cnt} 个")

        # 统计各类型关系数量
        relation_types = ["causes", "fixes", "depends_on", "monitors", "configures",
                         "relates_to", "indicates", "restarts", "checks"]
        total_relations = 0
        for rt in relation_types:
            result = session.run(f"MATCH ()-[r:{rt}]->() RETURN count(r) AS cnt")
            cnt = result.single()["cnt"]
            total_relations += cnt
            if cnt > 0:
                logger.info(f"  [{rt}]: {cnt} 条")

        logger.info(f"  总计: {total_entities} 个实体, {total_relations} 条关系")

        # 测试查询
        test_queries = ["Redis", "MySQL", "Kubernetes", "内存溢出", "CrashLoopBackOff"]
        logger.info("\n查询测试:")
        for q in test_queries:
            result = session.run(
                "MATCH (e {name: $name})-[r]-(n) "
                "RETURN e.name AS source, labels(e)[0] AS source_type, "
                "type(r) AS relation, n.name AS target, labels(n)[0] AS target_type "
                "LIMIT 5",
                name=q,
            )
            records = list(result)
            if records:
                logger.info(f"  '{q}' -> {len(records)} 条关联")
                for rec in records[:3]:
                    logger.info(f"    {rec['source']}({rec['source_type']}) --[{rec['relation']}]--> {rec['target']}({rec['target_type']})")
            else:
                logger.info(f"  '{q}' -> 无关联")

        return total_entities, total_relations


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SmartOps 知识图谱数据导入")
    parser.add_argument("--no-clear", action="store_true", help="不清空旧数据，追加导入")
    parser.add_argument("--uri", type=str, default=NEO4J_URI, help="Neo4j URI")
    parser.add_argument("--user", type=str, default=NEO4J_USER, help="Neo4j 用户名")
    parser.add_argument("--password", type=str, default=NEO4J_PASSWORD, help="Neo4j 密码")
    args = parser.parse_args()

    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.error("请先安装 neo4j: pip install neo4j")
        return

    logger.info(f"连接 Neo4j: {args.uri}")
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    driver.verify_connectivity()
    logger.info("Neo4j 连接成功")

    try:
        # 1. 清空旧数据
        if not args.no_clear:
            clear_neo4j(driver)

        # 2. 创建约束
        create_constraints(driver)

        # 3. 导入三元组
        logger.info(f"准备导入 {len(TRIPLES)} 条三元组...")
        entity_count, relation_count = import_triples(driver, TRIPLES)

        # 4. 验证
        total_entities, total_relations = verify_graph(driver)

        logger.info("\n" + "=" * 60)
        logger.info("知识图谱导入完成!")
        logger.info(f"  实体: {total_entities} 个")
        logger.info(f"  关系: {total_relations} 条")
        logger.info(f"  覆盖: Redis/MySQL/Nginx/K8s/Docker/Linux/Kafka/ES/RabbitMQ/MongoDB/Java/网络/微服务/监控")
        logger.info("=" * 60)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
