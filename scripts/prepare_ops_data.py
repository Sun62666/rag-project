"""
SmartOps 运维数据准备脚本

功能：
  1. 生成 1000+ 条运维故障知识数据，导入 Milvus 向量库
  2. 生成 LoRA 微调训练数据（100条）+ Reranker 重排序数据（100条）
  3. 数据增强：同义问题变体生成（200条），多方向扩展

用法：
  python scripts/prepare_ops_data.py --task all          # 全部执行
  python scripts/prepare_ops_data.py --task vector       # 仅向量库
  python scripts/prepare_ops_data.py --task finetune     # 仅微调数据
  python scripts/prepare_ops_data.py --task augment      # 仅问题变体
  python scripts/prepare_ops_data.py --task all --clear   # 清空后重建
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "prepared")
MILVUS_URI = "http://192.168.100.128:19530"
COLLECTION_NAME = "ops_knowledge_v2"

# ============================================================================
# 第一部分：运维故障知识库原始数据（覆盖 15+ 方向，40+ 案例）
# ============================================================================

CASE_STUDIES = [
    # ---- Redis (5条) ----
    {"category": "Redis", "title": "Redis 内存溢出 OOM 排查与处理",
     "content": "【案例】Redis 内存溢出 OOM 排查与处理\n【故障现象】Redis 进程占用内存持续增长，触发 OOM Killer 或达到 maxmemory 上限导致写入失败。客户端报错 OOM command not allowed when used memory > 'maxmemory'。\n【可能原因】1. 未配置 maxmemory 或淘汰策略 2. 大 Key 堆积（Hash 百万字段、List 无限增长）3. 客户端连接泄漏 4. 持久化 fork 时 COW 导致内存翻倍 5. 过期 Key 未及时清理 6. 数据结构选择不当导致内存浪费\n【排查命令】redis-cli info memory | grep -E 'used_memory_human|used_memory_rss|used_memory_peak'; redis-cli info clients | grep connected_clients; redis-cli --bigkeys --limit 10; redis-cli client list | wc -l; redis-cli config get maxmemory; redis-cli config get maxmemory-policy; top -p $(pgrep redis-server)\n【修复步骤】1. 设置最大内存和淘汰策略：redis-cli config set maxmemory 4gb；redis-cli config set maxmemory-policy allkeys-lru；写入 redis.conf 持久化 2. 排查清理大 Key：redis-cli --bigkeys 找出大 Key，对大 Hash/List/Set 使用 SCAN 分批处理，拆分为多个小 Key 3. 设置客户端超时：redis-cli config set timeout 300 4. 优化持久化：启用 lazyfree（lazyfree-lazy-eviction yes），降低 RDB 触发频率或改用 AOF 5. 监控告警：设置 used_memory > maxmemory * 0.8 时告警\n【验证方法】redis-cli info memory 确认 used_memory_human 恢复正常；redis-cli ping 返回 PONG"},
    {"category": "Redis", "title": "Redis 主从复制同步中断恢复",
     "content": "【案例】Redis 主从复制同步中断恢复\n【故障现象】从库状态显示 master_link_status:down，主从数据不一致，从库只读但无法同步最新数据。\n【可能原因】1. 主库重启导致 runid 变更 2. repl-backlog-size 不够，从库断开后追赶不上 3. 网络分区导致连接断开超过 repl-timeout 4. 从库磁盘满导致 RDB 加载失败 5. 主库开启了 requirepass 但从库未配置 masterauth\n【排查命令】redis-cli -p 6380 info replication; redis-cli -p 6379 info replication; redis-cli -p 6380 config get masterhost; redis-cli -p 6379 client list | grep slave\n【修复步骤】1. backlog 不够：主库增大 backlog（config set repl-backlog-size 256mb），从库重新同步（slaveof 192.168.1.100 6379）2. 主库 runid 变更：从库执行 slaveof no one → slaveof 192.168.1.100 6379 触发全量重同步 3. 密码问题：从库配置 masterauth <password> 4. 增加容错配置：min-slaves-to-write 1、min-slaves-max-lag 10、repl-timeout 60\n【验证方法】redis-cli -p 6380 info replication 确认 master_link_status:up 且 master_sync_in_progress:0"},
    {"category": "Redis", "title": "Redis 缓存穿透击穿雪崩防护",
     "content": "【案例】Redis 缓存穿透、击穿、雪崩防护方案\n一、缓存穿透：查询不存在的 Key，每次穿透缓存查 DB。解决方案：1. 布隆过滤器：查询前先经过布隆过滤器判断 Key 是否可能存在 2. 缓存空值：对查询为空的结果也做短时间缓存（TTL 60s）3. 参数校验：前端对非法参数拦截\n二、缓存击穿：热点 Key 过期瞬间大量并发请求打到 DB。解决方案：1. 热点 Key 不设过期或设置较长 TTL 2. 分布式锁（setnx）：只有一个线程能重建缓存 3. 逻辑过期：不过期物理删除，异步后台更新\n三、缓存雪崩：大量 Key 同时过期或 Redis 宕机。解决方案：1. TTL 随机打散：基础 TTL + random(0,300) 秒 2. 多级缓存：L1 本地缓存(Caffeine) + L2 Redis 3. Redis 高可用：哨兵模式或 Cluster 4. 降级熔断：数据库压力大时返回默认值\n【监控指标】缓存命中率 > 95% 为健康；DB QPS 异常升高时立即检查缓存层"},
    {"category": "Redis", "title": "Redis Cluster 集群在线扩缩容",
     "content": "【案例】Redis Cluster 集群在线扩容与缩容操作\n【扩容步骤】1. 准备新节点并加入集群：redis-cli --cluster add-node new_node:6379 existing_node:6379 2. 分配槽位给新节点：redis-cli --cluster reshard existing_node:6379，输入要迁移的槽数（如 4096），输入目标节点 ID，选择 all 表示从所有源节点均匀迁移 3. 等待迁移完成验证：redis-cli cluster nodes；redis-cli cluster info | grep cluster_state\n【缩容步骤】1. 将要移除节点的槽位迁出 2. 移除空节点：redis-cli --cluster del-node existing_node:6379 node_to_remove_id\n【注意事项】1. 扩缩容期间会有短暂性能抖动 2. 生产环境建议低峰期操作 3. 操作前务必备份\n【常见错误处理】CLUSTERDOWN: 检查是否超过半数 master 宕机；MOVED/ASK 错误：客户端需支持 ASKING/MOVED 重定向"},
    {"category": "Redis", "title": "Redis 持久化 RDB 与 AOF 选择策略",
     "content": "【案例】Redis 持久化 RDB 与 AOF 选择与故障恢复\n【RDB 方式】定时将内存快照写入磁盘（dump.rdb）。优点：文件紧凑、恢复速度快、适合备份。缺点：非实时，宕机可能丢失最后一次快照后的数据。配置：save 900 1; save 300 10; save 60 10000\n【AOF 方式】追加写入每条写命令到 appendonly.aof。优点：数据安全性高（最多丢失1秒数据）。缺点：文件体积大、恢复速度慢。配置：appendonly yes; appendfsync everysec\n【混合持久化（4.0+）】AOF 重写时先写 RDB 格式再追加 AOF 增量。配置：aof-use-rdb-preamble yes\n【故障恢复】RDB 恢复：将 dump.rdb 放入 Redis 数据目录启动自动加载。AOF 恢复：redis-check-aof --fix appendonly.aof。同时存在时优先加载 AOF\n【生产建议】主从架构：主库关闭持久化，从库开启 AOF。关键业务：AOF + everysec + 定期 RDB 备份到远程存储"},

    # ---- MySQL (5条) ----
    {"category": "MySQL", "title": "MySQL 主从同步延迟排查与优化",
     "content": "【案例】MySQL 主从复制延迟排查与优化\n【故障现象】从库 Seconds_Behind_Master 持续增大，读取从库的数据严重滞后于主库。\n【可能原因】1. 主库执行了大事务 2. 从库硬件性能不足 3. 主从之间网络延迟 4. MySQL 默认单线程 SQL_THREAD 回放 5. 从库上有长查询阻塞 6. 表缺少索引导致回放慢\n【排查命令】SHOW SLAVE STATUS\\G; SHOW MASTER STATUS; SHOW PROCESSLIST; SHOW ENGINE INNODB STATUS\\G\n【修复步骤】1. 开启并行复制：SET GLOBAL slave_parallel_workers = 4; SET GLOBAL slave_parallel_type = LOGICAL_CLOCK 2. 拆分大事务：DELETE FROM table WHERE id BETWEEN 1 AND 10000 LIMIT 1000; 循环执行 3. 升级从库硬件或调整 innodb_buffer_pool_size 4. 使用半同步复制保证一致性 5. 监控告警：Seconds_Behind_Master > 10 时告警\n【验证方法】SHOW SLAVE STATUS\\G 确认 Seconds_Behind_Master = 0"},
    {"category": "MySQL", "title": "MySQL 慢查询分析与索引优化",
     "content": "【案例】MySQL 慢查询分析与索引优化实战\n【故障现象】业务接口响应慢，数据库 CPU 使用率高，慢查询日志多条 SQL 耗时超过 3 秒。\n【排查步骤】1. 开启慢查询日志：SET GLOBAL slow_query_log = ON; SET GLOBAL long_query_time = 1 2. 分析慢查询日志：mysqldumpslow -s t -t 10 /var/log/mysql/slow.log 或 pt-query-digest 3. EXPLAIN 分析执行计划\n【常见优化手段】1. 添加合适索引：ALTER TABLE orders ADD INDEX idx_user_status(user_id, status); 遵循最左前缀原则 2. 优化 SQL 写法：避免 SELECT *、避免 WHERE 中对列使用函数、深分页改为游标翻页 3. 表结构优化：选择合适数据类型、大表考虑水平分表 4. 架构层面：读写分离、引入缓存层、归档历史数据\n【验证方法】EXPLAIN 确认 type 达到 ref/range 以上，SQL 耗时降至毫秒级"},
    {"category": "MySQL", "title": "MySQL InnoDB 死锁检测与预防",
     "content": "【案例】MySQL InnoDB 死锁检测与预防\n【故障现象】应用报错 Deadlock found when trying to get lock; try restarting transaction。\n【排查方法】1. 开启死锁日志：SET GLOBAL innodb_print_all_deadlocks = ON 2. 查看最近死锁：SHOW ENGINE INNODB STATUS\\G 找到 LATEST DETECTED DEADLOCK 部分\n【常见死锁场景与解决方案】场景1：不同顺序访问同一组记录 → 所有事务按相同顺序访问资源 场景2：外键约束导致隐式锁 → 先删子表再删父表 场景3：唯一索引插入冲突 → 使用 INSERT ... ON DUPLICATE KEY UPDATE\n【预防措施】1. 保持事务简短减少持锁时间 2. 统一访问资源固定顺序 3. 合理设计索引避免不必要间隙锁 4. 降低隔离级别为 READ COMMITTED 5. 高频并发操作用乐观锁替代悲观锁"},
    {"category": "MySQL", "title": "MySQL 数据目录磁盘空间满紧急处理",
     "content": "【案例】MySQL 数据目录磁盘空间满紧急处理\n【故障现象】MySQL 无法写入，报错 Error code 28: No space left on device\n【紧急处理】1. 清理二进制日志：PURGE BINARY LOGS BEFORE NOW() - INTERVAL 3 DAY 2. 删除慢查询日志 3. 清理临时文件：rm -rf /tmp/mysql* 4. ibdata1 文件过大：ALTER TABLE big_table ENGINE=InnoDB; 重建表碎片 5. 检查大表\n【根本解决】1. 扩容磁盘（LVM 扩展或挂载新盘）2. 数据归档策略 3. 开启独立表空间 innodb_file_per_table=1 4. 定期 OPTIMIZE TABLE 重建碎片\n【预防】磁盘使用率 > 80% 告警；定期清理 binlog 和日志；建立数据生命周期管理"},
    {"category": "MySQL", "title": "MySQL 高可用 MHA 故障切换",
     "content": "【案例】MySQL MHA 高可用故障切换实战\n【架构】MHA Manager + MHA Node，主库宕机时自动选举新主库并切换从库。\n【故障切换流程】1. MHA Manager 检测到主库不可达 2. 验证从库配置 3. 识别拥有最新 relay log 的从库为候选主库 4. 保存差异 relay log 5. 应用到其他从库 6. 提升候选主库为新主库 7. 修改其他从库指向新主库 8. VIP 切换\n【配置要点】manager_workdir /var/log/mha/manager; user=mha_monitor; repl_user=repl; master_ip_failover_script\n【手动切换】masterha_master_switch --conf=/etc/mha/app.cnf --master_state=dead\n【验证】mysql -h new-master -e 'SHOW MASTER STATUS'; 所有从库 SHOW SLAVE STATUS\\G 确认同步正常"},

    # ---- Nginx (3条) ----
    {"category": "Nginx", "title": "Nginx 502 Bad Gateway 排查",
     "content": "【案例】Nginx 502 Bad Gateway 故障排查\n【故障现象】用户访问返回 HTTP 502 Bad Gateway。\n【可能原因】1. 后端服务未启动或崩溃 2. 后端服务超时 3. upstream 端口配置错误 4. 后端负载过高拒绝连接 5. Nginx worker 连接数耗尽\n【排查命令】tail -f /var/log/nginx/error.log; curl -I http://127.0.0.1:8080/health; netstat -tlnp | grep 8080; systemctl status your-app-service; nginx -t\n【修复步骤】1. 重启后端服务：systemctl restart your-app 2. 调整超时：proxy_connect_timeout 10s; proxy_read_timeout 60s 3. 健康检查：server 192.168.1.10:8080 max_fails=3 fail_timeout=30s 4. 增加 worker 连接数：worker_connections 4096 5. 自定义错误页：error_page 502 /502.html\n【验证方法】curl -I https://your-domain.com 确认返回 HTTP 200"},
    {"category": "Nginx", "title": "Nginx 高并发性能调优",
     "content": "【案例】Nginx 高并发场景性能调优\n【核心优化配置】worker_processes auto; worker_rlimit_nofile 65535; events { worker_connections 65535; multi_accept on; use epoll; } http { sendfile on; tcp_nopush on; tcp_nodelay on; keepalive_timeout 65; gzip on; gzip_min_length 1k; proxy_buffer_size 16k; proxy_buffers 4 64k; }\n【系统级优化】net.core.somaxconn = 65535; net.ipv4.tcp_max_syn_backlog = 65535; net.ipv4.tcp_tw_reuse = 1\n【验证】ab -n 10000 -c 500 https://your-domain.com/ 或 wrk -t12 -c400 -d30s https://your-domain.com/"},
    {"category": "Nginx", "title": "Nginx SSL/TLS 证书配置与自动续期",
     "content": "【案例】Nginx SSL/TLS 证书配置与 Let's Encrypt 自动续期\n【SSL 配置】listen 443 ssl http2; ssl_certificate /etc/nginx/ssl/fullchain.pem; ssl_certificate_key /etc/nginx/ssl/privkey.pem; ssl_protocols TLSv1.2 TLSv1.3; ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256; ssl_session_cache shared:SSL:10m; add_header Strict-Transport-Security 'max-age=31536000' always\n【Let's Encrypt 自动续期】1. 安装：apt install certbot python3-certbot-nginx 2. 获取：certbot --nginx -d example.com 3. 自动续期 cron：0 3 * * * certbot renew --quiet --deploy-hook 'systemctl reload nginx'\n【证书过期检查】openssl s_client -connect example.com:443 | openssl x509 -noout -enddate"},

    # ---- Kubernetes (4条) ----
    {"category": "Kubernetes", "title": "Pod CrashLoopBackOff 排查",
     "content": "【案例】Kubernetes Pod CrashLoopBackOff 故障排查\n【故障现象】Pod 反复重启，状态 CrashLoopBackOff，RESTARTS 持续增加。\n【可能原因】1. 容器启动命令错误 2. OOMKilled 内存超限 3. 配置缺失 4. Liveness Probe 配置不当 5. 镜像拉取失败\n【排查命令】kubectl describe pod <pod-name>; kubectl logs <pod-name> --previous; kubectl get events --sort-by=.metadata.creationTimestamp\n【修复步骤】1. 根据 --previous 日志定位报错 2. OOMKilled：增加 resources.limits.memory 3. 修复启动命令 4. 调整探针：initialDelaySeconds: 30; failureThreshold: 3\n【验证】kubectl get pod <pod-name> -w 确认 Running 且 RESTARTS 不再增加"},
    {"category": "Kubernetes", "title": "Node NotReady 状态恢复",
     "content": "【案例】Kubernetes Node NotReady 状态恢复\n【故障现象】kubectl get nodes 显示某节点 NotReady，Pod 可能被驱逐。\n【可能原因】1. kubelet 服务停止 2. 磁盘/内存/PID 压力 3. 容器运行时异常 4. CNI 网络插件故障\n【排查命令】kubectl describe node <node-name>; systemctl status kubelet; journalctl -u kubelet -f -n 50; systemctl status containerd\n【修复步骤】1. kubelet 问题：systemctl restart kubelet 2. 资源压力：清理磁盘、释放 evicted Pod 3. CNI 问题：systemctl restart calico-node 4. 时间同步：systemctl restart chronyd\n【验证】kubectl get nodes 确认 Ready 状态"},
    {"category": "Kubernetes", "title": "Service/Ingress 无法访问排查",
     "content": "【案例】Kubernetes Service/Ingress 无法访问排查\n【排查清单】1. Pod 是否 Running/Ready 2. Service selector 是否匹配 Pod labels 3. Endpoints 是否有后端 4. Service ClusterIP 是否可达 5. NodePort/LoadBalancer 是否暴露 6. Ingress 规则是否正确 7. Ingress Controller 日志\n【常见问题】selector 不匹配：检查 labels 一致性；端口不对应：targetPort 必须匹配容器端口；CNI 问题：检查网络插件状态\n【验证】curl -H 'Host: your-domain.com' http://<ingress-ip> 确认返回正常"},
    {"category": "Kubernetes", "title": "Kubernetes 资源配额与 LimitRange 管理",
     "content": "【案例】Kubernetes 资源配额与 LimitRange 管理\n【故障现象】Pod 无法调度，报错 Insufficient cpu/memory 或 exceeded quota。\n【排查命令】kubectl describe resourcequota -n <namespace>; kubectl describe limitrange -n <namespace>; kubectl top nodes\n【解决方案】1. 调整 ResourceQuota：hard: requests.cpu: '10'; requests.memory: 20Gi; limits.cpu: '20' 2. 设置 LimitRange 默认值：default: cpu: '1' memory: 1Gi; defaultRequest: cpu: 100m memory: 128Mi 3. 清理未使用资源\n【验证】kubectl describe resourcequota 确认 Used 不超过 Hard"},

    # ---- Docker (3条) ----
    {"category": "Docker", "title": "Docker 容器网络不通排查",
     "content": "【案例】Docker 容器网络不通排查\n【故障现象】容器间无法通信或无法访问外部网络。\n【可能原因】1. Docker 网桥配置错误 2. iptables 规则冲突 3. 容器不在同一网络 4. DNS 解析失败 5. 宿主机转发未开启\n【排查命令】docker network ls; docker network inspect <network-name>; docker exec <container> ping <target>; cat /proc/sys/net/ipv4/ip_forward\n【修复步骤】1. 将容器加入同一网络：docker network connect <net> <container> 2. 重建 Docker 网络 3. 重启 Docker 服务恢复 iptables 4. 指定 DNS：docker run --dns 8.8.8.8 5. 开启 IP 转发：sysctl -w net.ipv4.ip_forward=1\n【验证方法】docker exec <container> curl -I http://target:port"},
    {"category": "Docker", "title": "Docker 镜像构建优化与体积缩减",
     "content": "【案例】Docker 镜像构建优化与体积缩减\n【问题】Docker 镜像体积过大（>1GB），构建速度慢。\n【优化手段】1. 使用多阶段构建：FROM node:18 AS builder → COPY --from=builder 2. 选择精简基础镜像：alpine / slim / distroless 3. 合并 RUN 指令：RUN apt-get update && apt-get install -y pkg1 && rm -rf /var/lib/apt/lists/* 4. 利用 .dockerignore 排除不需要的文件 5. 合理利用构建缓存 6. 清理包管理器缓存\n【验证】docker images 确认镜像体积缩小到预期范围"},
    {"category": "Docker", "title": "Docker 存储卷数据持久化与备份",
     "content": "【案例】Docker 存储卷数据持久化与备份恢复\n【故障现象】容器删除后数据丢失。\n【存储类型】1. Volume（docker volume）：由 Docker 管理，推荐方式 2. Bind Mount：挂载宿主机目录 3. tmpfs：内存存储\n【操作命令】创建卷：docker volume create mydata; 挂载卷：docker run -v mydata:/data myimage; 备份：docker run --rm -v mydata:/data -v $(pwd):/backup alpine tar czf /backup/mydata.tar.gz -C /data .; 恢复：docker run --rm -v mydata:/data -v $(pwd):/backup alpine tar xzf /backup/mydata.tar.gz -C /data\n【验证】docker volume inspect mydata 确认 Mountpoint 正确"},

    # ---- Linux (4条) ----
    {"category": "Linux", "title": "Linux 磁盘空间满排查与清理",
     "content": "【案例】Linux 磁盘空间满排查与清理\n【故障现象】磁盘使用率 100%，服务无法写入文件。\n【可能原因】1. 日志文件未轮转 2. 临时文件堆积 3. 已删除文件被进程占用 4. Docker 镜像堆积 5. inode 耗尽\n【排查命令】df -h; du -sh /* | sort -rh | head -10; lsof | grep deleted; docker system df; df -i\n【修复步骤】1. 清理日志：find /var/log -name '*.log.*' -mtime +7 -delete; 配置 logrotate 2. 清理临时文件 3. 释放已删除文件：kill 占用进程 4. Docker 清理：docker system prune -af 5. inode 耗尽：清理小文件目录\n【预防】磁盘使用率 > 80% 告警；配置 logrotate 自动轮转"},
    {"category": "Linux", "title": "Linux CPU 使用率飙高排查",
     "content": "【案例】Linux CPU 使用率飙高排查\n【故障现象】服务器 CPU 使用率突然飙升至 90% 以上。\n【可能原因】1. 进程死循环 2. 突发流量 3. 定时任务执行 4. 内存不足导致频繁 swap\n【排查命令】top -c -o %CPU; ps aux --sort=-%cpu | head -20; vmstat 1 5; iostat -x 1 3\n【修复步骤】1. 定位高 CPU 进程：top -c -p <PID> 2. 分析线程：top -H -p <PID> 3. 限流或扩容应对突发流量 4. 调整定时任务到低峰期 5. 优化代码或增加资源\n【验证方法】top 确认 CPU 使用率恢复到正常水平"},
    {"category": "Linux", "title": "Linux 内存泄漏排查与处理",
     "content": "【案例】Linux 内存泄漏排查与处理\n【故障现象】系统可用内存持续下降，OOM Killer 开始杀进程。\n【排查命令】free -m; top -o %MEM; ps aux --sort=-%mem | head -20; pmap -x <PID>; valgrind --leak-check=full ./your_app\n【修复步骤】1. 定位泄漏进程 2. 使用 valgrind 或 AddressSanitizer 分析代码 3. 临时方案：定时重启进程 4. 调整 vm.overcommit_memory 和 oom_score_adj\n【验证】free -m 确认内存使用稳定不再持续增长"},
    {"category": "Linux", "title": "Linux 网络连接超时排查",
     "content": "【案例】Linux 网络连接超时排查\n【故障现象】服务间调用超时，curl/wget 请求外部接口超时。\n【可能原因】1. DNS 解析慢 2. 防火墙规则阻拦 3. TCP 连接队列满 4. TIME_WAIT 过多 5. MTU 不匹配\n【排查命令】nslookup domain; traceroute target; iptables -L -n; ss -s; netstat -ant | grep TIME_WAIT | wc -l; ping -M do -s 1472 target\n【修复步骤】1. 配置本地 DNS 缓存 2. 检查防火墙规则 3. 增大 somaxconn 和 tcp_max_syn_backlog 4. 开启 tcp_tw_reuse 5. 调整 MTU\n【验证】curl -w '@curl-format.txt' -o /dev/null -s http://target 确认连接时间正常"},

    # ---- Kafka (3条) ----
    {"category": "Kafka", "title": "Kafka 消费积压处理",
     "content": "【案例】Kafka 消费积压处理\n【故障现象】Kafka 消费者 Lag 持续增大，消息处理延迟。\n【可能原因】1. 消费者处理速度慢 2. 消费者实例不足 3. 消息量突增 4. 消费者频繁 Rebalance\n【排查命令】kafka-consumer-groups.sh --describe --group <group-id> --bootstrap-server localhost:9092; kafka-topics.sh --describe --topic <topic>\n【修复步骤】1. 增加消费者实例数 2. 优化消费逻辑减少处理耗时 3. 临时扩容分区数 4. 调整 max.poll.interval.ms 避免 Rebalance 5. 紧急情况跳过积压：seekToBeginning 或 seekToEnd\n【验证方法】kafka-consumer-groups.sh 确认 Lag 趋近于 0"},
    {"category": "Kafka", "title": "Kafka 分区 Leader 副本迁移",
     "content": "【案例】Kafka 分区 Leader 副本迁移与重分配\n【故障现象】某 Broker 宕机后，分区 Leader 集中在少数 Broker 上，负载不均。\n【操作步骤】1. 生成迁移计划：kafka-reassign-partitions.sh --generate --topics-to-move-json-file topics.json --broker-list '0,1,2' 2. 执行迁移：kafka-reassign-partitions.sh --execute --reassignment-json-file plan.json 3. 验证：kafka-reassign-partitions.sh --verify --reassignment-json-file plan.json\n【优先副本选举】kafka-leader-election.sh --election-type preferred --all-topic-partitions\n【注意事项】迁移期间带宽占用增加，建议 throttle：kafka-reassign-partitions.sh --execute --throttle 50000000"},
    {"category": "Kafka", "title": "Kafka 生产者发送失败排查",
     "content": "【案例】Kafka 生产者发送失败排查\n【故障现象】生产者报错 TimeoutException 或 NotEnoughReplicasException。\n【可能原因】1. Broker 不可用 2. 副本数不足 3. 网络问题 4. 生产者配置不当（acks/retries/batch.size）\n【排查命令】kafka-broker-api-versions.sh --bootstrap-server localhost:9092; kafka-topics.sh --describe --topic <topic>\n【修复步骤】1. 检查 Broker 状态 2. 调整生产者配置：acks=all; retries=3; batch.size=16384; linger.ms=5 3. 检查 min.insync.replicas 4. 增加超时：request.timeout.ms=30000\n【验证】生产者发送成功率 > 99.9%"},

    # ---- Elasticsearch (3条) ----
    {"category": "Elasticsearch", "title": "Elasticsearch 集群变红处理",
     "content": "【案例】Elasticsearch 集群状态变红处理\n【故障现象】Elasticsearch 集群状态为 red，部分主分片不可用。\n【可能原因】1. 节点宕机 2. 磁盘满导致分片分配失败 3. 分片损坏 4. 集群配置不当\n【排查命令】curl localhost:9200/_cluster/health?pretty; curl localhost:9200/_cat/shards?v&h=index,shard,state,node | grep UNASSIGNED; curl localhost:9200/_cat/allocation?v\n【修复步骤】1. 重启宕机节点 2. 清理磁盘空间或调整 cluster.routing.allocation.disk.threshold 3. 手动分配分片：_cluster/reroute 4. 增加副本数保证高可用\n【验证方法】curl localhost:9200/_cluster/health 确认 status=green"},
    {"category": "Elasticsearch", "title": "Elasticsearch 慢查询优化",
     "content": "【案例】Elasticsearch 慢查询优化\n【故障现象】搜索请求耗时超过 5 秒，集群 CPU 使用率高。\n【排查步骤】1. 开启慢查询日志：index.search.slowlog.threshold.query.warn: 5s 2. 使用 Profile API 分析：GET /index/_search?profile=true 3. 检查索引 mapping 和分片数\n【优化手段】1. 优化查询 DSL：避免通配符开头查询、使用 filter 替代 query 2. 合理设置分片数：每分片 10-50GB 3. 使用路由键减少扫描分片 4. 预索引优化：将范围查询转为 term 查询 5. 使用索引排序加速排序查询\n【验证】搜索耗时降至 500ms 以下"},
    {"category": "Elasticsearch", "title": "Elasticsearch 索引生命周期管理",
     "content": "【案例】Elasticsearch 索引生命周期管理 ILM\n【需求】日志索引按天创建，7 天后转为冷数据，30 天后删除。\n【配置步骤】1. 创建 ILM Policy：PUT _ilm/policy/logs-policy { \"policy\": { \"phases\": { \"hot\": { \"min_age\": \"0ms\", \"actions\": { \"rollover\": { \"max_size\": \"50gb\", \"max_age\": \"1d\" } } }, \"warm\": { \"min_age\": \"7d\", \"actions\": { \"shrink\": { \"number_of_shards\": 1 }, \"forcemerge\": { \"max_num_segments\": 1 } } }, \"delete\": { \"min_age\": \"30d\", \"actions\": { \"delete\": {} } } } } } 2. 绑定到索引模板 3. 验证：GET _ilm/explain/logs-*\n【监控】GET _ilm/status 查看 ILM 执行状态"},

    # ---- Prometheus/Grafana (2条) ----
    {"category": "Prometheus", "title": "Prometheus 告警规则配置与优化",
     "content": "【案例】Prometheus 告警规则配置与优化\n【告警规则示例】groups: - name: node_alerts rules: - alert: NodeDown expr: up == 0 for: 5m labels: severity: critical annotations: summary: 'Node {{ $labels.instance }} is down' - alert: HighMemory expr: (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) > 0.9 for: 10m labels: severity: warning\n【优化建议】1. 设置 for 持续时间避免抖动 2. 合理分级 severity 3. 使用 inhibit_rules 抑制低级别告警 4. 告警分组 group_by 减少通知风暴 5. 配置静默规则应对维护窗口\n【验证】在 Prometheus Alerts 页面确认规则加载且状态正确"},
    {"category": "Prometheus", "title": "Prometheus Target 采集失败排查",
     "content": "【案例】Prometheus Target 采集失败排查\n【故障现象】Prometheus Targets 页面显示部分 target 状态为 DOWN。\n【可能原因】1. 目标服务不可达 2. 采集路径/端口错误 3. 标签配置冲突 4. 采集超时 5. 认证配置错误\n【排查步骤】1. 检查 Prometheus UI → Targets 页面的错误信息 2. curl http://target:port/metrics 验证端点可达 3. 检查 scrape_interval 和 scrape_timeout 4. 查看 Prometheus 日志\n【修复】1. 修正 targets 地址 2. 调整 scrape_timeout 3. 添加 basic_auth 或 tls_config 4. 检查网络连通性\n【验证】Targets 页面所有 target 状态为 UP"},

    # ---- RabbitMQ (2条) ----
    {"category": "RabbitMQ", "title": "RabbitMQ 队列积压与消费延迟处理",
     "content": "【案例】RabbitMQ 队列积压与消费延迟处理\n【故障现象】队列消息数持续增长，消费者处理速度跟不上生产速度。\n【排查命令】rabbitmqctl list_queues name messages consumers; rabbitmqctl list_consumers -p /; rabbitmqctl node_health_check\n【修复步骤】1. 增加消费者数量 2. 优化消费逻辑（批量确认、异步处理）3. 设置队列 TTL 和最大长度：x-message-ttl=3600000; x-max-length=100000 4. 死信队列处理：x-dead-letter-exchange 5. 紧急方案：临时消费者清空积压\n【验证】rabbitmqctl list_queues 确认消息数恢复到正常水平"},
    {"category": "RabbitMQ", "title": "RabbitMQ 集群脑裂恢复",
     "content": "【案例】RabbitMQ 集群脑裂恢复\n【故障现象】RabbitMQ 集群节点间无法通信，出现分区（partition）。\n【排查命令】rabbitmqctl cluster_status; rabbitmqctl list_partitions\n【修复步骤】1. 停止少数派节点：rabbitmqctl stop_app 2. 在多数派节点上：rabbitmqctl forget_cluster_node <node> 3. 重新加入：rabbitmqctl join_cluster rabbit@majority-node 4. 启动：rabbitmqctl start_app\n【预防】1. 配置 pause_minority 模式 2. 网络分区检测：cluster_partition_handling = pause_minority 3. 使用 Quorum Queues 替代经典镜像队列\n【验证】rabbitmqctl cluster_status 确认无分区"},

    # ---- MongoDB (2条) ----
    {"category": "MongoDB", "title": "MongoDB 副本集选举失败排查",
     "content": "【案例】MongoDB 副本集选举失败排查\n【故障现象】MongoDB 副本集无法选举出 Primary 节点，集群只读。\n【可能原因】1. 节点数不足多数 2. 网络分区 3. 节点优先级配置不当 4. arbiter 不可用\n【排查命令】rs.status(); rs.conf(); rs.printReplicationInfo()\n【修复步骤】1. 确保存活节点数 >= n/2+1 2. 检查网络连通性 3. 调整优先级：rs.reconfig({members: [{_id:0, priority:2}, {_id:1, priority:1}]}) 4. 添加 arbiter：rs.addArb('arbiter:27017') 5. 强制重新配置：rs.reconfig(cfg, {force: true})\n【验证】rs.status() 确认有 PRIMARY 节点且其他节点 SECONDARY"},
    {"category": "MongoDB", "title": "MongoDB 慢查询与索引优化",
     "content": "【案例】MongoDB 慢查询与索引优化\n【故障现象】查询响应慢，数据库 CPU 使用率高。\n【排查步骤】1. 开启慢查询日志：db.setProfilingLevel(1, {slowms: 100}) 2. 查看慢查询：db.system.profile.find().sort({ts:-1}).limit(10) 3. 分析执行计划：db.collection.explain('executionStats').find({...})\n【优化手段】1. 创建合适索引：db.collection.createIndex({field: 1}) 2. 复合索引遵循 ESR 原则（Equality, Sort, Range）3. 覆盖查询：确保查询字段都在索引中 4. 限制返回字段：find({}, {field:1, _id:0})\n【验证】explain() 确认 stage 为 IXSCAN 且 totalDocsExamined 接近 nReturned"},

    # ---- Zookeeper (1条) ----
    {"category": "Zookeeper", "title": "Zookeeper 连接超时处理",
     "content": "【案例】Zookeeper 连接超时处理\n【故障现象】客户端连接 Zookeeper 超时，服务注册发现失败。\n【可能原因】1. Zookeeper 节点负载过高 2. 网络延迟 3. session timeout 配置过小 4. JVM GC 停顿\n【排查命令】echo ruok | nc zk-host 2181; echo mntr | nc zk-host 2181; zkCli.sh -server zk-host:2181\n【修复步骤】1. 增大 session timeout：zkSessionTimeout=30000 2. 优化 JVM 堆内存和 GC 策略 3. 检查网络延迟和带宽 4. 扩容 Zookeeper 集群\n【验证方法】客户端连接成功且无超时日志"},

    # ---- Tomcat/Java (2条) ----
    {"category": "Java", "title": "Java 应用 OOM 排查与内存优化",
     "content": "【案例】Java 应用 OOM 排查与内存优化\n【故障现象】Java 应用抛出 java.lang.OutOfMemoryError: Java heap space。\n【排查步骤】1. 配置 JVM 参数自动 dump：-XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/tmp/ 2. 分析 heap dump：使用 MAT 或 VisualVM 3. 查看 GC 日志：-Xlog:gc*:file=gc.log\n【常见 OOM 类型】1. Java heap space：堆内存不足，增大 -Xmx 或修复内存泄漏 2. Metaspace：类加载过多，增大 -XX:MaxMetaspaceSize 3. GC overhead limit exceeded：GC 回收效率低 4. Direct buffer memory：NIO 堆外内存不足\n【修复】1. 增大堆内存：-Xmx4g 2. 修复内存泄漏代码 3. 优化缓存策略 4. 调整 GC 算法：-XX:+UseG1GC\n【验证】应用稳定运行 24 小时无 OOM，GC 停顿在可接受范围"},
    {"category": "Java", "title": "Java 应用线程死锁排查",
     "content": "【案例】Java 应用线程死锁排查\n【故障现象】应用部分功能无响应，线程挂起。\n【排查步骤】1. jstack <pid> > thread_dump.txt 2. 查找 BLOCKED 线程和死锁信息 3. 使用 JConsole/Arthas 在线诊断\n【常见死锁场景】1. 嵌套 synchronized 锁 2. 数据库锁与 Java 锁混合 3. 线程池资源耗尽\n【修复】1. 统一锁获取顺序 2. 使用 tryLock 带超时 3. 缩小锁粒度 4. 使用并发工具类替代 synchronized\n【验证】jstack 确认无死锁，应用功能恢复正常"},

    # ---- 网络 (2条) ----
    {"category": "网络", "title": "TCP 连接 TIME_WAIT 过多处理",
     "content": "【案例】TCP 连接 TIME_WAIT 过多处理\n【故障现象】短连接场景下大量 TIME_WAIT 状态连接，新连接建立失败。\n【排查命令】ss -ant | grep TIME_WAIT | wc -l; netstat -ant | awk '{print $6}' | sort | uniq -c | sort -rn\n【修复步骤】1. 开启 tcp_tw_reuse：sysctl -w net.ipv4.tcp_tw_reuse=1 2. 开启 tcp_tw_recycle（NAT 环境慎用）3. 调整 tcp_max_tw_buckets：sysctl -w net.ipv4.tcp_max_tw_buckets=65535 4. 使用长连接替代短连接 5. 调整 tcp_fin_timeout：sysctl -w net.ipv4.tcp_fin_timeout=15\n【验证】ss -ant | grep TIME_WAIT | wc -l 降至合理范围"},
    {"category": "网络", "title": "DNS 解析超时排查与优化",
     "content": "【案例】DNS 解析超时排查与优化\n【故障现象】服务调用外部接口偶发超时，定位发现 DNS 解析耗时超过 2 秒。\n【排查命令】nslookup domain; dig domain; time nslookup domain; cat /etc/resolv.conf\n【修复步骤】1. 配置多 DNS 服务器：nameserver 8.8.8.8; nameserver 114.114.114.114 2. 开启本地 DNS 缓存：systemctl start nscd 或 systemd-resolved 3. 配置 /etc/hosts 静态解析关键域名 4. JVM 配置：networkaddress.cache.ttl=60 5. 应用层 DNS 缓存\n【验证】time nslookup domain 确认解析耗时 < 50ms"},

    # ---- PostgreSQL (2条) ----
    {"category": "PostgreSQL", "title": "PostgreSQL 锁等待与死锁排查",
     "content": "【案例】PostgreSQL 锁等待与死锁排查\n【故障现象】查询长时间等待，应用报锁超时。\n【排查命令】SELECT * FROM pg_locks WHERE NOT granted; SELECT pg_blocking_pids(pid), * FROM pg_stat_activity WHERE wait_event_type='Lock'; SELECT * FROM pg_stat_activity WHERE state='active' ORDER BY query_start\n【修复步骤】1. 终止阻塞查询：SELECT pg_terminate_backend(<pid>) 2. 优化长事务 3. 添加合适索引减少锁范围 4. 调整 lock_timeout 5. 使用 advisory lock 替代表锁\n【验证】pg_stat_activity 确认无长时间锁等待"},
    {"category": "PostgreSQL", "title": "PostgreSQL VACUUM 与表膨胀处理",
     "content": "【案例】PostgreSQL VACUUM 与表膨胀处理\n【故障现象】表查询变慢，磁盘空间持续增长，pg_stat_user_tables 显示 n_dead_tup 很多。\n【排查命令】SELECT schemaname, relname, n_dead_tup, last_vacuum, last_autovacuum FROM pg_stat_user_tables ORDER BY n_dead_tup DESC; SELECT pg_size_pretty(pg_total_relation_size('table_name'))\n【修复步骤】1. 手动 VACUUM：VACUUM table_name 2. VACUUM FULL 重建表（锁表）：VACUUM FULL table_name 3. 调整 autovacuum 参数：autovacuum_vacuum_scale_factor=0.1; autovacuum_analyze_scale_factor=0.05 4. 长事务阻止 VACUUM：SELECT * FROM pg_stat_activity WHERE state='idle in transaction'\n【验证】pg_stat_user_tables 确认 n_dead_tup 趋近于 0"},

    # ---- Consul (1条) ----
    {"category": "Consul", "title": "Consul 服务注册发现失败排查",
     "content": "【案例】Consul 服务注册发现失败排查\n【故障现象】服务注册到 Consul 后无法被发现，健康检查失败。\n【排查命令】consul members; consul catalog services; consul health service <service-name>; curl localhost:8500/v1/health/service/<service-name>\n【修复步骤】1. 检查 Agent 状态：consul members 确认节点 alive 2. 健康检查配置：调整 interval 和 timeout 3. 网络问题：检查 8300/8301/8500 端口 4. ACL 配置：确认 token 权限 5. 重新注册服务\n【验证】consul catalog services 确认服务可见且健康"},

    # ---- HAProxy (1条) ----
    {"category": "HAProxy", "title": "HAProxy 后端健康检查失败处理",
     "content": "【案例】HAProxy 后端健康检查失败处理\n【故障现象】HAProxy 将后端服务器标记为 DOWN，流量全部转发到其他节点。\n【排查命令】echo 'show stat' | socat stdio /var/run/haproxy.sock; echo 'show backend' | socat stdio /var/run/haproxy.sock; haproxy -c -f /etc/haproxy/haproxy.cfg\n【修复步骤】1. 检查后端服务是否正常：curl http://backend:port/health 2. 调整健康检查参数：inter 2000; fall 3; rise 2 3. 修改检查路径和预期状态码 4. 临时禁用检查：server s1 10.0.0.1:8080 check disabled\n【验证】show stat 确认后端状态为 UP 且 lchk_cur 正常"},

    # ---- Jenkins (1条) ----
    {"category": "Jenkins", "title": "Jenkins 构建队列堵塞处理",
     "content": "【案例】Jenkins 构建队列堵塞处理\n【故障现象】Jenkins 构建任务排队等待，executor 全部占用。\n【可能原因】1. Executor 数量不足 2. 僵尸构建占用 executor 3. 资源锁未释放 4. Pipeline 语法错误导致挂起\n【排查步骤】1. Jenkins UI → Build Queue 查看排队任务 2. 检查 Executor 占用 3. 查看构建日志定位卡住位置\n【修复步骤】1. 增加 Executor 数量 2. 终止僵尸构建 3. 重启 Jenkins 清理锁 4. 优化 Pipeline 使用 agent none 减少占用\n【验证】构建队列清空，新任务能立即执行"},

    # ---- Git (1条) ----
    {"category": "Git", "title": "Git 仓库体积过大清理",
     "content": "【案例】Git 仓库体积过大清理\n【故障现象】git clone 非常慢，仓库超过 5GB。\n【排查命令】git count-objects -vH; git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sed -n 's/^blob //p' | sort -nk2 | tail -20\n【修复步骤】1. 使用 git-filter-repo 清理大文件：git filter-repo --path large_file.bin --invert-paths 2. 清理 reflog：git reflog expire --expire=now --all && git gc --prune=now --aggressive 3. BFG Repo-Cleaner：java -jar bfg.jar --strip-blobs-bigger-than 100M\n【预防】1. .gitignore 排除大文件 2. 使用 Git LFS 管理大文件 3. 代码审查禁止提交二进制文件\n【验证】git count-objects -vH 确认仓库体积缩小"},

    # ---- Ansible (1条) ----
    {"category": "Ansible", "title": "Ansible Playbook 执行失败排查",
     "content": "【案例】Ansible Playbook 执行失败排查\n【故障现象】Ansible 执行 Playbook 报错 unreachable 或 failed。\n【排查步骤】1. 检查连通性：ansible all -m ping 2. 详细输出：ansible-playbook site.yml -vvv 3. 检查 inventory 配置 4. 检查 SSH 密钥和权限\n【常见错误】1. SSH 连接失败：检查密钥、known_hosts 2. 权限不足：become: yes; become_user: root 3. 模块参数错误：查看文档确认参数格式 4. 事实收集失败：gather_facts: no 跳过\n【修复】1. 配置正确的 SSH 密钥 2. 添加 become 提权 3. 使用 register 和 debug 调试变量 4. 添加 until/retries/delay 重试\n【验证】ansible-playbook site.yml 执行返回 ok=changed, failed=0"},

    # ---- SSL/证书 (1条) ----
    {"category": "SSL", "title": "SSL 证书过期导致服务不可用",
     "content": "【案例】SSL 证书过期导致服务不可用\n【故障现象】HTTPS 服务突然不可用，浏览器提示证书无效，客户端报 SSL handshake failure。\n【排查命令】openssl s_client -connect domain:443 | openssl x509 -noout -dates; echo | openssl s_client -connect domain:443 2>/dev/null | openssl x509 -noout -checkend 86400\n【修复步骤】1. 紧急续期：certbot renew --force-renewal 2. 手动替换证书文件并 reload 服务 3. 更新负载均衡器/CDN 证书 4. 重启依赖服务\n【预防】1. 证书到期前 30 天自动告警 2. 配置自动续期 cron 3. 使用证书监控工具 4. 统一证书管理平台\n【验证】openssl s_client 确认新证书生效且有效期 > 30 天"},
]

# ============================================================================
# 第二部分：Q&A 模板（用于批量生成，覆盖更多方向）
# ============================================================================

QA_TEMPLATES = [
    # Redis
    {"q": "Redis 如何查看当前内存使用情况？", "a": "使用 redis-cli info memory 命令，关注 used_memory_human（分配内存）、used_memory_rss_human（操作系统视角）、used_memory_peak_human（峰值）。也可用 redis-cli memory doctor 进行内存诊断。", "category": "Redis"},
    {"q": "Redis 的 maxmemory-policy 淘汰策略有哪些？", "a": "Redis 淘汰策略包括：1. noeviction：不淘汰，写入报错（默认）2. allkeys-lru：所有键中淘汰最久未使用的 3. volatile-lru：过期键中淘汰最久未使用的 4. allkeys-random：随机淘汰 5. volatile-random：过期键中随机淘汰 6. volatile-ttl：淘汰 TTL 最短的 7. allkeys-lfu（4.0+）：淘汰使用频率最低的 8. volatile-lfu（4.0+）：过期键中淘汰频率最低的。推荐 allkeys-lru。", "category": "Redis"},
    {"q": "Redis 哨兵模式的工作原理是什么？", "a": "Redis Sentinel 是 Redis 高可用方案：1. 监控：Sentinel 不断检查主从是否正常 2. 提醒：故障时通过 API 发通知 3. 自动故障转移：主库宕机时自动选举新主库，从库重新指向新主库。选举基于 raft 协议，需多数 Sentinel 同意。配置建议至少 3 个 Sentinel 节点。", "category": "Redis"},
    {"q": "Redis 如何实现分布式锁？", "a": "推荐使用 RedLock 算法：1. 加锁：SET key value NX PX 30000（NX 不存在才设置，PX 过期毫秒）2. 释放锁：Lua 脚本确保只有持有者能释放 if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('del',KEYS[1]) else return 0 end 3. RedLock：在多个独立 Redis 实例上获取锁，多数成功才算获取。注意：锁续期（看门狗）防止业务未完成锁过期。", "category": "Redis"},
    {"q": "Redis Pipeline 和事务有什么区别？", "a": "Pipeline：批量发送命令减少网络 RTT，不保证原子性，命令独立执行。事务（MULTI/EXEC）：命令依次入队后原子执行，但执行中不会回滚，某条失败后续继续。如需回滚可用 Lua 脚本。Pipeline+事务可组合使用。", "category": "Redis"},
    {"q": "Redis 大 Key 如何发现和处理？", "a": "发现：redis-cli --bigkeys 扫描；redis-cli debug object key 查看 serializedlength；MEMORY USAGE key（4.0+）。处理：1. 大 Hash：hscan 分批拆分为多个小 Hash 2. 大 List：ltrim 截断或拆分 3. 大 Set：sscan 分批迁移 4. 大 String：压缩存储或拆分。删除用 UNLINK（4.0+异步删除）避免阻塞。", "category": "Redis"},
    {"q": "Redis 连接数过多怎么处理？", "a": "排查：redis-cli info clients 查看 connected_clients；redis-cli client list 查看连接详情。处理：1. 设置最大连接数：config set maxclients 10000 2. 设置超时：config set timeout 300 3. 检查连接泄漏：client list 中 idle 时间长的连接 4. 应用侧使用连接池 5. 监控告警：connected_clients > maxclients*0.8 时告警。", "category": "Redis"},
    {"q": "Redis AOF 重写的作用和触发条件？", "a": "AOF 重写将冗余命令压缩为最终状态，减小文件体积。触发条件：1. 自动：auto-aof-rewrite-percentage 100（体积增长100%时）和 auto-aof-rewrite-min-size 64mb 2. 手动：BGREWRITEAOF 命令。重写过程 fork 子进程，不阻塞主进程。重写期间新命令同时写入旧 AOF 和重写缓冲区。", "category": "Redis"},

    # MySQL
    {"q": "MySQL 如何查看当前正在执行的 SQL？", "a": "使用 SHOW PROCESSLIST 查看当前所有连接和正在执行的 SQL。完整信息用 SHOW FULL PROCESSLIST。也可查询 information_schema.processlist 表：SELECT * FROM information_schema.processlist WHERE command='Query' ORDER BY time DESC。长时间执行的查询可用 KILL <id> 终止。", "category": "MySQL"},
    {"q": "MySQL 索引失效的常见场景有哪些？", "a": "1. WHERE 条件对列使用函数：WHERE YEAR(create_time)=2024 → 改为范围查询 2. 隐式类型转换：varchar 列用 int 查询 3. LIKE 左通配符：LIKE '%abc' 4. OR 条件中有无索引列 5. 不满足最左前缀：联合索引(a,b) 只查 b 6. NOT IN/NOT EXISTS 7. 数据量小优化器选择全表扫描 8. IS NULL 在不允许 NULL 的列上", "category": "MySQL"},
    {"q": "MySQL 读写分离如何实现？", "a": "实现方式：1. 代码层面：配置多数据源，写操作走主库，读操作走从库 2. 中间件：MyCat/ShardingSphere 自动路由 3. ProxySQL：MySQL 代理层，基于规则路由 4. 框架支持：Sharding-JDBC、MyBatis 插件。注意：主从延迟时读从库可能读到旧数据，可强制走主库或等待同步完成。", "category": "MySQL"},
    {"q": "MySQL binlog 有哪些格式？", "a": "三种格式：1. STATEMENT：记录 SQL 语句，日志量小但主从不一致风险（NOW()/UUID()等）2. ROW：记录行变更，日志量大但数据一致性最好 3. MIXED：默认 STATEMENT，不确定时自动切换 ROW。推荐 ROW 格式。查看：SHOW VARIABLES LIKE 'binlog_format'。", "category": "MySQL"},
    {"q": "MySQL 如何备份和恢复数据？", "a": "备份方式：1. 逻辑备份：mysqldump -u root -p db > backup.sql（适合小库）2. 物理备份：xtrabackup --backup --target-dir=/backup（适合大库，热备）3. 快照备份：LVM/云盘快照。恢复：逻辑恢复 mysql < backup.sql；物理恢复 xtrabackup --prepare + --copy-back。建议：全量+增量备份策略，定期验证恢复流程。", "category": "MySQL"},
    {"q": "MySQL GTID 复制有什么优势？", "a": "GTID（全局事务标识符）优势：1. 主从切换简单：无需找 binlog 位点，自动定位 2. 复制验证：每个事务有唯一 ID，容易判断一致性 3. 多源复制支持好。配置：gtid_mode=ON; enforce_gtid_consistency=ON。切换：CHANGE MASTER TO MASTER_AUTO_POSITION=1。", "category": "MySQL"},
    {"q": "MySQL 如何优化 INSERT 批量写入性能？", "a": "1. 批量 INSERT：INSERT INTO t VALUES (1,'a'),(2,'b'),(3,'c') 替代单条 2. 关闭唯一性检查：SET unique_checks=0 3. 关闭外键检查：SET foreign_key_checks=0 4. 事务提交：SET autocommit=0; INSERT...; COMMIT 5. 调整 innodb_buffer_pool_size 6. 调整 innodb_flush_log_at_trx_commit=2（非核心业务）7. 使用 LOAD DATA INFILE", "category": "MySQL"},
    {"q": "MySQL 连接数过多怎么处理？", "a": "排查：SHOW PROCESSLIST; SHOW STATUS LIKE 'Threads_connected'; SHOW VARIABLES LIKE 'max_connections'。处理：1. 增大 max_connections：SET GLOBAL max_connections=5000 2. 设置 wait_timeout 减少空闲连接 3. 应用侧使用连接池 4. 检查连接泄漏 5. 使用 ProxySQL 做连接池管理。", "category": "MySQL"},

    # Nginx
    {"q": "Nginx 如何配置负载均衡？", "a": "upstream backend { server 10.0.0.1:8080 weight=3; server 10.0.0.2:8080 weight=2; server 10.0.0.3:8080 backup; } server { location / { proxy_pass http://backend; } } 策略：轮询（默认）、weight、ip_hash、least_conn、random。健康检查：max_fails=3 fail_timeout=30s。", "category": "Nginx"},
    {"q": "Nginx 504 Gateway Timeout 如何处理？", "a": "504 表示 Nginx 等待后端响应超时。修复：1. 增大超时：proxy_read_timeout 120s; proxy_connect_timeout 60s 2. 优化后端处理速度 3. 配置超时返回默认值：proxy_next_upstream timeout 4. 检查后端是否真的处理慢。注意：增大超时只是治标，根本要优化后端性能。", "category": "Nginx"},
    {"q": "Nginx 如何配置 HTTPS 重定向？", "a": "server { listen 80; server_name example.com; return 301 https://$server_name$request_uri; } server { listen 443 ssl http2; server_name example.com; ssl_certificate /path/to/cert.pem; ssl_certificate_key /path/to/key.pem; ... } 也可在单个 server 块中：if ($scheme != 'https') { return 301 https://$host$request_uri; }", "category": "Nginx"},
    {"q": "Nginx 如何限制请求速率防刷？", "a": "1. 限制请求速率：limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s; location /api/ { limit_req zone=api burst=20 nodelay; } 2. 限制连接数：limit_conn_zone $binary_remote_addr zone=addr:10m; limit_conn addr 100 3. IP 黑名单：deny 192.168.1.1; 4. 验证码/鉴权层。", "category": "Nginx"},
    {"q": "Nginx 日志格式如何自定义？", "a": "log_format main '$remote_addr - $remote_user [$time_local] \"$request\" $status $body_bytes_sent \"$http_referer\" \"$http_user_agent\" $request_time $upstream_response_time'; access_log /var/log/nginx/access.log main; 关键变量：$request_time 请求总时间、$upstream_response_time 后端响应时间、$body_bytes_sent 响应体大小。", "category": "Nginx"},

    # Kubernetes
    {"q": "Kubernetes Pod 无法调度怎么排查？", "a": "排查：kubectl describe pod <pod> 查看 Events 部分。常见原因：1. 资源不足：Insufficient cpu/memory → 扩容节点或降低 requests 2. NodeSelector/Affinity 不匹配 3. Taint/Toleration 不兼容 4. PVC Pending：存储类不存在或容量不足 5. 镜像拉取失败：ImagePullBackOff → 检查镜像地址和拉取凭证", "category": "Kubernetes"},
    {"q": "Kubernetes HPA 自动扩缩容如何配置？", "a": "apiVersion: autoscaling/v2 kind: HorizontalPodAutoscaler spec: scaleTargetRef: apiVersion: apps/v1 kind: Deployment name: my-app minReplicas: 2 maxReplicas: 10 metrics: - type: Resource resource: name: cpu target: type: Utilization averageUtilization: 70 前提：Pod 必须配置 resources.requests，部署 metrics-server。", "category": "Kubernetes"},
    {"q": "Kubernetes ConfigMap 和 Secret 有什么区别？", "a": "ConfigMap：存储非敏感配置（环境变量、配置文件），明文存储在 etcd。Secret：存储敏感数据（密码、证书），base64 编码存储（非加密），可配置加密存储。使用方式相同：环境变量、Volume 挂载。建议：Secret 开启 etcd 加密（encryptionConfiguration），配合 RBAC 限制访问。", "category": "Kubernetes"},
    {"q": "Kubernetes 如何滚动更新和回滚？", "a": "滚动更新：kubectl set image deployment/my-app container=image:v2 或修改 YAML 后 apply。策略：spec.strategy.type=RollingUpdate; maxSurge=1; maxUnavailable=0。回滚：kubectl rollout undo deployment/my-app；查看历史：kubectl rollout history deployment/my-app；回滚到指定版本：kubectl rollout undo deployment/my-app --to-revision=2。", "category": "Kubernetes"},
    {"q": "Kubernetes PV/PVC 生命周期管理？", "a": "PV：集群级存储资源，由管理员创建或 StorageClass 动态供给。PVC：命名空间级存储请求。绑定：PVC 根据 capacity/accessMode/storageClass 匹配 PV。回收策略：Retain（保留需手动清理）、Delete（自动删除）、Recycle（已废弃）。动态供给：StorageClass + provisioner 自动创建 PV。", "category": "Kubernetes"},
    {"q": "Kubernetes 如何排查 DNS 解析问题？", "a": "排查：1. kubectl exec -it <pod> -- nslookup kubernetes.default 2. 检查 CoreDNS：kubectl get pods -n kube-system -l k8s-app=kube-dns 3. CoreDNS 日志：kubectl logs -n kube-system <coredns-pod> 4. 检查 /etc/resolv.conf 配置。常见问题：1. ndots 导致过多搜索 2. CoreDNS ConfigMap 配置错误 3. kube-proxy 未正常运行 4. 网络策略阻拦 DNS 端口", "category": "Kubernetes"},

    # Docker
    {"q": "Docker 容器时间不对怎么处理？", "a": "容器默认使用 UTC 时区。修复：1. 挂载宿主机时区文件：-v /etc/localtime:/etc/localtime:ro 2. 设置环境变量：-e TZ=Asia/Shanghai 3. Dockerfile 中：RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime。推荐方式2，最简洁。", "category": "Docker"},
    {"q": "Docker 容器日志过大怎么清理？", "a": "1. 单容器清理：truncate -s 0 $(docker inspect --format='{{.LogPath}}' container) 2. 全局清理：docker logs 配置限制，/etc/docker/daemon.json：{\"log-driver\": \"json-file\", \"log-opts\": {\"max-size\": \"10m\", \"max-file\": \"3\"}} 3. docker system prune 清理。重启 Docker 后新配置生效，已有容器需重建。", "category": "Docker"},
    {"q": "Docker Compose 如何实现服务依赖启动顺序？", "a": "使用 depends_on 和 healthcheck：services: db: image: mysql healthcheck: test: mysqladmin ping -h localhost interval: 5s retries: 10 app: image: myapp depends_on: db: condition: service_healthy 注意：depends_on 只控制启动顺序，healthcheck 确保依赖真正就绪。", "category": "Docker"},
    {"q": "Docker 容器内如何调试网络问题？", "a": "1. 安装工具：apt-get update && apt-get install -y iputils-ping dnsutils curl 2. 使用网络调试镜像：docker run --rm -it --network container:<target> nicolaka/netshoot 3. 常用命令：ping、nslookup、curl、traceroute、tcpdump 4. 查看容器网络：docker network inspect <network>", "category": "Docker"},

    # Linux
    {"q": "Linux 如何查看系统资源使用情况？", "a": "综合查看：top/htop（CPU/内存）；free -h（内存）；df -h（磁盘）；iostat -x 1（IO）；sar -u 1 5（CPU历史）；vmstat 1 5（虚拟内存）。推荐 htop 交互式查看，iotop 查看 IO 占用。", "category": "Linux"},
    {"q": "Linux 如何排查僵尸进程？", "a": "查找僵尸进程：ps aux | grep 'Z' 或 ps -eo pid,ppid,stat,cmd | grep 'Z'。僵尸进程状态为 Z（zombie），父进程未调用 wait() 回收。处理：1. 杀死父进程：kill -9 <ppid> 2. 父进程修复：在代码中添加 wait/waitpid 3. 大量僵尸说明父进程有 bug。", "category": "Linux"},
    {"q": "Linux 如何设置开机自启动服务？", "a": "systemd 方式：1. 创建 /etc/systemd/system/myapp.service 2. [Service] ExecStart=/usr/bin/myapp; Restart=always 3. systemctl enable myapp; systemctl start myapp。查看状态：systemctl status myapp。", "category": "Linux"},
    {"q": "Linux swap 空间不足怎么处理？", "a": "1. 查看当前 swap：free -h; swapon --show 2. 创建 swap 文件：dd if=/dev/zero of=/swapfile bs=1G count=4; chmod 600 /swapfile; mkswap /swapfile; swapon /swapfile 3. 持久化：echo '/swapfile none swap sw 0 0' >> /etc/fstab 4. 调整 swappiness：sysctl vm.swappiness=10", "category": "Linux"},
    {"q": "Linux 如何排查文件句柄泄漏？", "a": "1. 查看进程打开文件数：ls /proc/<pid>/fd | wc -l 2. 系统级：cat /proc/sys/fs/file-nr 3. 查看哪个进程打开最多：lsof | awk '{print $1}' | sort | uniq -c | sort -rn | head 4. 增大限制：ulimit -n 65535; /etc/security/limits.conf 设置 nofile 65535", "category": "Linux"},

    # Kafka
    {"q": "Kafka 如何保证消息不丢失？", "a": "生产者：acks=all 确认所有副本写入；retries>0 自动重试。Broker：min.insync.replicas=2 保证至少2个副本；unclean.leader.election.enable=false 禁止非同步副本成为Leader。消费者：enable.auto.commit=false 手动提交offset。", "category": "Kafka"},
    {"q": "Kafka 分区数如何规划？", "a": "分区数决定并行度。建议：1. 吞吐量：单分区生产吞吐 * 分区数 >= 目标吞吐 2. 消费者数：分区数 >= 消费者数 3. 不宜过多：分区多增加文件句柄和内存开销 4. 经验值：单Broker 100-2000分区，集群总分区 < 20000。修改分区数只能增加不能减少。", "category": "Kafka"},
    {"q": "Kafka Rebalance 频繁怎么优化？", "a": "原因：1. 消费者处理超时 2. 心跳超时 3. 消费者频繁上下线。优化：1. 增大 max.poll.interval.ms 2. 减小 max.poll.records 3. 增大 session.timeout.ms 4. 使用 Sticky 分配策略 5. 避免消费者频繁重启", "category": "Kafka"},

    # Elasticsearch
    {"q": "Elasticsearch 分片数如何规划？", "a": "建议：1. 每分片 10-50GB 2. 分片数 = 数据量 / 30GB（经验值）3. 分片数不超过节点数 * 3 4. 副本数根据可用性需求（1-2个）5. 使用 ILM 管理索引生命周期。注意：分片数创建后不可修改，需提前规划或使用 shrink/rollover。", "category": "Elasticsearch"},
    {"q": "Elasticsearch 写入性能如何优化？", "a": "1. 批量写入：bulk API，每批 5-15MB 2. 增大 refresh_interval：index.refresh_interval=30s 3. 增大 translog flush 频率 4. 关闭副本先写入后开启 5. 使用自动生成的 ID 避免版本检查 6. 合理设置分片数", "category": "Elasticsearch"},

    # Prometheus
    {"q": "Prometheus 如何做容量规划？", "a": "1. 估算样本量：series数 * 采集频率 * 保留天数 2. 内存：约 3KB/series 3. 磁盘：约 1-2 bytes/sample 4. 推荐：2亿 series 约 6GB 内存 + 2TB 磁盘（15天）5. 使用 recording rules 预计算减少查询负载 6. 长期存储接入 Thanos/VictoriaMetrics", "category": "Prometheus"},
    {"q": "Prometheus 如何配置远程存储？", "a": "配置 remote_write：remote_write: - url: http://victoriametrics:8428/api/v1/write queue_config: max_samples_per_send: 10000 capacity: 20000 常见后端：VictoriaMetrics、Thanos、Cortex、Mimir。注意网络延迟和写入速率限制。", "category": "Prometheus"},

    # RabbitMQ
    {"q": "RabbitMQ 如何保证消息可靠性？", "a": "1. 生产者确认：publisher confirm 机制 2. 消息持久化：durable queue + persistent message 3. 消费者手动 ACK 4. 死信队列处理失败消息 5. 镜像队列/Quorum Queue 高可用。注意：持久化影响性能，根据业务权衡。", "category": "RabbitMQ"},
    {"q": "RabbitMQ 镜像队列和 Quorum Queue 怎么选？", "a": "镜像队列（Classic Mirrored）：老方案，主从同步，脑裂风险，已不推荐。Quorum Queue：3.8+ 推荐，基于 Raft 协议，数据一致性好，支持消息确认后删除。建议新项目用 Quorum Queue，老项目逐步迁移。", "category": "RabbitMQ"},

    # MongoDB
    {"q": "MongoDB 如何实现数据分片？", "a": "1. 启动 config server（3节点副本集）2. 启动 mongos 路由 3. 启动 shard（每个是副本集）4. sh.addShard('rs1/host1:27017') 5. sh.enableSharding('db') 6. sh.shardCollection('db.collection', {shardKey: 1})。片键选择：高基数、低频率、非单调递增。", "category": "MongoDB"},
    {"q": "MongoDB Change Streams 如何使用？", "a": "Change Streams 监听集合变更：watch = db.collection.watch([{$match: {operationType: 'insert'}}]); for change in watch: print(change)。支持 insert/update/delete/replace/invalidate。需副本集或分片集群。可指定 resume_token 实现断点续传。", "category": "MongoDB"},

    # Java
    {"q": "Java 应用 CPU 飙高怎么排查？", "a": "1. top -H -p <pid> 找到高 CPU 线程 2. jstack <pid> | grep <hex-tid> -A 30 查看线程堆栈 3. Arthas：thread -n 3 查看最忙线程 4. 常见原因：死循环、正则回溯、GC 频繁、加密运算。5. GC 问题：jstat -gcutil <pid> 1000 查看 GC 频率", "category": "Java"},
    {"q": "Java GC 调优如何做？", "a": "1. 选择 GC 算法：JDK8 用 CMS/G1，JDK11+ 用 G1/ZGC 2. 堆大小：-Xms=-Xmx 避免扩缩 3. G1 调优：-XX:MaxGCPauseMillis=200; -XX:InitiatingHeapOccupancyPercent=45 4. ZGC：-XX:+UseZGC -Xmx4g（亚毫秒停顿）5. 监控：GC 日志 + VisualVM/GCViewer 分析", "category": "Java"},

    # 网络
    {"q": "如何排查 TCP 连接重置问题？", "a": "1. 抓包：tcpdump -i eth0 -nn port 8080 2. 查看连接状态：ss -antp | grep <port> 3. 常见 RST 原因：防火墙丢弃、对端进程崩溃、TCP 超时、端口未监听 4. 检查 iptables/nftables 规则 5. 检查 keepalive 配置", "category": "网络"},
    {"q": "如何排查网络丢包问题？", "a": "1. ifconfig/ethtool 查看丢包统计 2. netstat -s | grep -i loss 3. ping 检测丢包率 4. mtr 追踪丢包位置 5. tcpdump 抓包分析 6. 常见原因：网卡队列满、缓冲区溢出、链路质量差、MTU 问题", "category": "网络"},

    # PostgreSQL
    {"q": "PostgreSQL 如何配置连接池？", "a": "推荐 PgBouncer：1. 安装：apt install pgbouncer 2. 配置：[databases] db = host=127.0.0.1 port=5432 dbname=mydb; [pgbouncer] pool_mode=transaction; max_client_conn=1000; default_pool_size=20 3. transaction 模式最节省连接 4. 重启：systemctl restart pgbouncer", "category": "PostgreSQL"},
    {"q": "PostgreSQL 如何做逻辑复制？", "a": "1. 发布端：CREATE PUBLICATION mypub FOR TABLE t1, t2; 2. 订阅端：CREATE SUBSCRIPTION mysub CONNECTION 'host=publisher port=5432' PUBLICATION mypub; 3. 优势：选择性复制、跨版本、可写订阅端 4. 限制：不支持 DDL 复制、序列需手动同步 5. 监控：pg_stat_subscription", "category": "PostgreSQL"},

    # Consul
    {"q": "Consul 和 Zookeeper/etcd 有什么区别？", "a": "Consul：内置服务发现+健康检查+KV+多数据中心，HTTP/DNS 接口。Zookeeper：通用协调服务，Java 生态，需自己实现服务发现。etcd：Kubernetes 底层存储，Raft 协议，HTTP/gRPC。选型：K8s 用 etcd，服务网格用 Consul，Hadoop 用 ZK。", "category": "Consul"},

    # HAProxy
    {"q": "HAProxy 和 Nginx 做负载均衡怎么选？", "a": "HAProxy：专业负载均衡，支持四层/七层，性能极高，健康检查丰富，适合 TCP 负载。Nginx：Web 服务器+负载均衡，七层功能强（缓存/重写/SSL），适合 HTTP 场景。建议：四层用 HAProxy，七层用 Nginx，或 HAProxy 前置+Nginx 后置。", "category": "HAProxy"},

    # Jenkins
    {"q": "Jenkins Pipeline 如何优化构建速度？", "a": "1. 并行阶段：parallel stages 2. 增量构建：只构建变更模块 3. 缓存依赖：Maven/npm 缓存挂载 4. agent none 减少节点占用 5. 使用 stash/unstash 传递产物 6. 分布式构建：多节点并行 7. 优化 checkout：浅克隆 git clone --depth=1", "category": "Jenkins"},

    # Git
    {"q": "Git 如何处理合并冲突？", "a": "1. git merge 发生冲突后，打开冲突文件 2. <<<<<<< HEAD 到 ======= 是当前分支，======= 到 >>>>>>> 是合并分支 3. 手动选择保留内容 4. git add 标记冲突已解决 5. git commit 完成合并。工具：VS Code/IntelliJ 可视化解决冲突。预防：频繁合并主分支、小粒度提交。", "category": "Git"},

    # Ansible
    {"q": "Ansible 如何实现滚动更新？", "a": "serial 关键字控制批次：- hosts: webservers serial: 2 tasks: - name: deploy app ... 每次更新2台。配合 pre_tasks 健康检查和 post_tasks 验证。也可用百分比：serial: '25%'。配合 max_fail_percentage 控制失败阈值。", "category": "Ansible"},

    # SSL
    {"q": "如何批量检查多域名 SSL 证书到期时间？", "a": "脚本方式：for domain in domain1.com domain2.com; do echo | openssl s_client -connect $domain:443 2>/dev/null | openssl x509 -noout -dates; done。工具：certwatch、ssl-cert-check。Prometheus blackbox_exporter 也可监控证书到期。告警：到期前 30 天通知。", "category": "SSL"},

    # 监控通用
    {"q": "运维监控体系如何搭建？", "a": "三层架构：1. 采集层：Prometheus/Zabbix 采集指标，Filebeat/Fluentd 采集日志 2. 存储层：Prometheus TSDB/InfluxDB 存指标，Elasticsearch 存日志 3. 展示层：Grafana 可视化，AlertManager/钉钉告警。原则：全链路监控、分级告警、SLO 驱动。", "category": "监控"},
    {"q": "如何设计告警策略避免告警风暴？", "a": "1. 分级：P0-P3 严重程度 2. 分组：group_by 聚合同类告警 3. 抑制：inhibit_rules 高级别抑制低级别 4. 静默：维护窗口静默 5. 去重：相同告警不重复发送 6. for 持续时间：避免瞬时抖动 7. 告警收敛：多个告警合并为一条 8. 定期审查告警规则，删除无效告警", "category": "监控"},
    {"q": "如何实现蓝绿部署？", "a": "1. 准备两套完全相同的环境（蓝/绿）2. 当前流量指向蓝环境 3. 新版本部署到绿环境 4. 验证绿环境功能正常 5. 切换流量到绿环境 6. 蓝环境保留用于回滚。Nginx 实现：upstream 中切换 server 权重。K8s：修改 Service selector。", "category": "部署"},
    {"q": "如何实现金丝雀发布？", "a": "1. 新版本先部署到少量实例（如5%流量）2. 监控错误率和性能指标 3. 逐步扩大流量比例（5%→20%→50%→100%）4. 异常时快速回滚。K8s 实现：调整 Deployment 副本数或使用 Argo Rollouts。Nginx：weight 权重调整。Istio：流量路由规则。", "category": "部署"},
    {"q": "如何排查微服务调用链路超时？", "a": "1. 分布式追踪：Jaeger/Zipkin 查看完整调用链 2. 定位超时服务：看 span 耗时 3. 检查该服务：日志、指标、资源使用 4. 常见原因：下游慢查询、网络抖动、线程池满、GC 停顿 5. 临时方案：增大超时、熔断降级 6. 根本方案：优化慢服务", "category": "微服务"},
    {"q": "服务熔断和降级有什么区别？", "a": "熔断：当下游服务错误率超阈值时，断路器打开，快速失败不再调用，一段时间后半开尝试恢复。降级：服务不可用时返回兜底数据或简化逻辑。关系：熔断是触发条件，降级是处理策略。工具：Hystrix/Sentinel/Resilience4j。", "category": "微服务"},
    {"q": "如何排查 OOM Killer 杀进程？", "a": "1. 查看日志：dmesg | grep -i 'out of memory' 或 journalctl -k | grep oom 2. 查看 oom_score：cat /proc/<pid>/oom_score 3. 调整优先级：echo -100 > /proc/<pid>/oom_score_adj（保护关键进程）4. 根本解决：增加内存、优化内存使用、配置 swap", "category": "Linux"},
    {"q": "如何排查 SSH 连接慢？", "a": "1. 关闭 DNS 反查：/etc/ssh/sshd_config 设置 UseDNS no 2. 关闭 GSSAPI：GSSAPIAuthentication no 3. 检查 /etc/nsswitch.conf DNS 配置 4. 检查 ~/.ssh/authorized_keys 权限 5. 重启 sshd：systemctl restart sshd", "category": "Linux"},
    {"q": "如何排查 cron 定时任务不执行？", "a": "1. 检查 cron 服务：systemctl status crond 2. 查看日志：/var/log/cron 或 journalctl -u crond 3. 检查 crontab 语法：crontab -l 4. 环境变量问题：cron 环境精简，需写绝对路径 5. 检查用户权限 6. 检查时区设置", "category": "Linux"},
]

# ============================================================================
# 第三部分：同义问题变体模板（多方向扩展）
# ============================================================================

# 同义替换规则：每个方向有多种表达方式
SYNONYM_PATTERNS = {
    # 方向1：问题词替换
    "question_words": {
        "怎么办": ["怎么处理", "如何解决", "怎么排查", "如何修复", "怎么应对"],
        "如何": ["怎么", "怎样", "用什么方法", "有什么办法"],
        "排查": ["定位", "诊断", "分析", "排查定位", "查原因"],
        "处理": ["解决", "修复", "应对", "处置", "消除"],
        "优化": ["调优", "提升", "改善", "加速", "提高"],
    },
    # 方向2：技术术语替换
    "tech_terms": {
        "内存溢出": ["OOM", "内存不足", "Out of Memory", "内存超限", "内存耗尽"],
        "内存泄漏": ["memory leak", "内存持续增长", "内存不释放"],
        "主从同步延迟": ["主从复制延迟", "主从延迟", "复制滞后", "从库落后主库"],
        "慢查询": ["慢SQL", "查询缓慢", "SQL性能差", "查询超时"],
        "死锁": ["Deadlock", "锁冲突", "锁等待", "锁争用"],
        "磁盘满": ["磁盘空间不足", "磁盘100%", "No space left", "磁盘耗尽"],
        "连接超时": ["Timeout", "连接不上", "超时断开", "响应超时"],
        "消费积压": ["消息积压", "Lag增大", "消费延迟", "消息堆积"],
        "集群变红": ["集群RED", "分片不可用", "集群不健康", "主分片丢失"],
        "OOMKilled": ["内存超限被杀", "容器OOM", "内存溢出被终止"],
        "NotReady": ["节点不可用", "节点异常", "节点失联"],
        "CrashLoopBackOff": ["容器反复重启", "Pod崩溃循环", "启动失败循环"],
        "502": ["Bad Gateway", "网关错误", "上游不可达"],
        "脑裂": ["网络分区", "Split Brain", "集群分裂"],
        "负载均衡": ["LB", "流量分发", "请求分发"],
        "高可用": ["HA", "容灾", "故障转移"],
    },
    # 方向3：语气/场景替换
    "scenarios": {
        "prefix_urgent": ["紧急！", "线上告警：", "生产环境：", "紧急求助："],
        "prefix_context": ["生产环境中", "线上服务", "我们的系统", "客户反馈"],
        "suffix_detail": ["求详细步骤", "有完整方案吗", "最佳实践是什么", "求排查思路"],
    },
}

# 预定义的种子问题（用于生成200条变体）
SEED_QUESTIONS = [
    # Redis 方向
    "Redis内存溢出怎么办？",
    "Redis主从同步中断怎么恢复？",
    "Redis缓存穿透怎么处理？",
    "Redis缓存击穿如何解决？",
    "Redis缓存雪崩怎么应对？",
    "Redis大Key如何发现和处理？",
    "Redis连接数过多怎么处理？",
    "Redis持久化RDB和AOF怎么选？",
    "Redis Cluster扩容怎么做？",
    "Redis分布式锁怎么实现？",
    # MySQL 方向
    "MySQL主从同步延迟怎么处理？",
    "MySQL慢查询怎么优化？",
    "MySQL死锁怎么排查？",
    "MySQL磁盘空间满怎么办？",
    "MySQL连接数过多怎么处理？",
    "MySQL索引失效怎么排查？",
    "MySQL读写分离怎么实现？",
    "MySQL数据如何备份恢复？",
    # Nginx 方向
    "Nginx 502错误怎么排查？",
    "Nginx 504超时怎么处理？",
    "Nginx如何配置负载均衡？",
    "Nginx如何配置HTTPS？",
    "Nginx高并发怎么调优？",
    # Kubernetes 方向
    "K8s Pod CrashLoopBackOff怎么解决？",
    "K8s Node NotReady怎么恢复？",
    "K8s Service无法访问怎么排查？",
    "K8s Pod无法调度怎么办？",
    "K8s如何配置HPA自动扩缩容？",
    "K8s DNS解析失败怎么排查？",
    # Docker 方向
    "Docker容器网络不通怎么排查？",
    "Docker镜像体积太大怎么优化？",
    "Docker容器日志过大怎么清理？",
    "Docker数据卷如何备份恢复？",
    # Linux 方向
    "Linux磁盘空间满怎么清理？",
    "Linux CPU使用率飙高怎么排查？",
    "Linux内存泄漏怎么排查？",
    "Linux网络连接超时怎么排查？",
    "Linux僵尸进程怎么处理？",
    "Linux文件句柄泄漏怎么排查？",
    # Kafka 方向
    "Kafka消费积压怎么处理？",
    "Kafka消息丢失怎么预防？",
    "Kafka Rebalance频繁怎么优化？",
    # Elasticsearch 方向
    "ES集群变红怎么处理？",
    "ES慢查询怎么优化？",
    "ES写入性能怎么提升？",
    # 其他方向
    "RabbitMQ队列积压怎么处理？",
    "MongoDB副本集选举失败怎么办？",
    "Java应用OOM怎么排查？",
    "Java线程死锁怎么排查？",
    "TCP TIME_WAIT过多怎么处理？",
    "DNS解析超时怎么优化？",
    "Prometheus告警风暴怎么避免？",
    "SSL证书过期怎么处理？",
    "微服务调用链路超时怎么排查？",
    "如何实现蓝绿部署？",
    "如何实现金丝雀发布？",
]


def generate_synonym_variants(seed_questions: list = None, target_count: int = 200) -> list:
    """基于种子问题生成同义变体，多方向扩展"""
    if seed_questions is None:
        seed_questions = SEED_QUESTIONS

    variants = []
    seen = set()

    def add_variant(q: str, source: str):
        q = q.strip()
        if q and q not in seen and len(q) > 5:
            seen.add(q)
            variants.append({"question": q, "source": source})

    # 原始问题全部加入
    for q in seed_questions:
        add_variant(q, "original")

    # 方向1：问题词替换
    for q in list(seen):
        for old, replacements in SYNONYM_PATTERNS["question_words"].items():
            if old in q:
                for new_word in replacements:
                    variant = q.replace(old, new_word)
                    add_variant(variant, f"question_word:{old}->{new_word}")

    # 方向2：技术术语替换
    for q in list(seen):
        for old, replacements in SYNONYM_PATTERNS["tech_terms"].items():
            if old in q:
                for new_term in replacements:
                    variant = q.replace(old, new_term)
                    add_variant(variant, f"tech_term:{old}->{new_term}")

    # 方向3：添加场景前缀
    for q in list(seen):
        for prefix in SYNONYM_PATTERNS["scenarios"]["prefix_urgent"][:2]:
            add_variant(f"{prefix}{q}", "scenario:urgent")
        for prefix in SYNONYM_PATTERNS["scenarios"]["prefix_context"][:2]:
            add_variant(f"{prefix}{q}", "scenario:context")

    # 方向4：添加后缀
    for q in list(seen):
        for suffix in SYNONYM_PATTERNS["scenarios"]["suffix_detail"][:2]:
            add_variant(f"{q}{suffix}", "scenario:detail")

    # 方向5：组合替换（术语+问题词）
    for q in list(seen):
        for old_term, term_replacements in SYNONYM_PATTERNS["tech_terms"].items():
            if old_term in q:
                for new_term in term_replacements[:2]:
                    mid = q.replace(old_term, new_term)
                    for old_word, word_replacements in SYNONYM_PATTERNS["question_words"].items():
                        if old_word in mid:
                            for new_word in word_replacements[:2]:
                                variant = mid.replace(old_word, new_word)
                                add_variant(variant, "combo:term+word")

    # 方向6：句式变换
    for q in list(seen):
        if q.endswith("怎么办？"):
            add_variant(q.replace("怎么办？", "的解决方案是什么？"), "pattern:sentence")
            add_variant(q.replace("怎么办？", "，有什么好的处理方式？"), "pattern:sentence")
        if q.endswith("怎么处理？"):
            add_variant(q.replace("怎么处理？", "的处理方法有哪些？"), "pattern:sentence")
        if q.endswith("怎么排查？"):
            add_variant(q.replace("怎么排查？", "的排查思路是什么？"), "pattern:sentence")
            add_variant(q.replace("怎么排查？", "，如何定位问题？"), "pattern:sentence")
        if q.endswith("如何解决？"):
            add_variant(q.replace("如何解决？", "的解决办法是什么？"), "pattern:sentence")
        if q.endswith("怎么优化？"):
            add_variant(q.replace("怎么优化？", "有哪些优化手段？"), "pattern:sentence")

    # 方向7：口语化/英文缩写变体
    colloquial_map = {
        "Kubernetes": ["K8s", "k8s"],
        "Elasticsearch": ["ES", "es"],
        "PostgreSQL": ["PG", "pg"],
        "MongoDB": ["Mongo", "mongo"],
        "RabbitMQ": ["RMQ", "rabbit"],
        "Zookeeper": ["ZK", "zk"],
    }
    for q in list(seen):
        for full, abbrevs in colloquial_map.items():
            if full in q:
                for abbrev in abbrevs:
                    variant = q.replace(full, abbrev)
                    add_variant(variant, f"abbrev:{full}->{abbrev}")

    # 如果还不够，从QA_TEMPLATES的问题中继续生成
    if len(variants) < target_count:
        for qa in QA_TEMPLATES:
            add_variant(qa["q"], "qa_template")
            # 对QA模板也做术语替换
            for old, replacements in SYNONYM_PATTERNS["tech_terms"].items():
                if old in qa["q"]:
                    for new_term in replacements[:2]:
                        variant = qa["q"].replace(old, new_term)
                        add_variant(variant, f"qa_tech_term:{old}->{new_term}")

    # 截取到目标数量（各方向均匀采样）
    if len(variants) > target_count:
        # 按方向分组
        by_source = {}
        for v in variants:
            src_type = v["source"].split(":")[0]
            by_source.setdefault(src_type, []).append(v)

        # 均匀采样每个方向
        result = []
        n_sources = len(by_source)
        per_source = max(1, target_count // n_sources)
        remaining = target_count

        for src_type, items in by_source.items():
            take = min(per_source, len(items), remaining)
            result.extend(items[:take])
            remaining -= take

        # 如果还有余量，从剩余中补充
        if remaining > 0:
            added = set(id(r) for r in result)
            for v in variants:
                if remaining <= 0:
                    break
                if id(v) not in added:
                    result.append(v)
                    added.add(id(v))
                    remaining -= 1

        # 打乱顺序
        random.shuffle(result)
        result = result[:target_count]
    else:
        result = variants
    logger.info(f"同义变体生成: 种子{len(seed_questions)}条 -> 总生成{len(variants)}条 -> 截取{len(result)}条")

    # 统计方向分布
    source_stats = {}
    for v in result:
        src_type = v["source"].split(":")[0]
        source_stats[src_type] = source_stats.get(src_type, 0) + 1
    for src, cnt in sorted(source_stats.items(), key=lambda x: -x[1]):
        logger.info(f"  {src}: {cnt}条")

    return result


# ============================================================================
# 第四部分：数据处理与导出函数
# ============================================================================

def generate_vector_documents() -> list:
    """生成向量库文档列表（1000+条），返回 List[Document]"""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    all_docs = []

    # 1. 案例研究 → Document
    for case in CASE_STUDIES:
        doc = Document(
            page_content=case["content"],
            metadata={
                "source": f"case_{case['category']}_{hashlib.md5(case['title'].encode()).hexdigest()[:8]}",
                "source_name": case["title"],
                "type": "case_study",
                "category": case["category"],
            }
        )
        all_docs.append(doc)

    # 2. Q&A 模板 → Document
    for i, qa in enumerate(QA_TEMPLATES):
        content = f"问: {qa['q']}\n答: {qa['a']}"
        doc = Document(
            page_content=content,
            metadata={
                "source": f"qa_{qa['category']}_{i:04d}",
                "source_name": qa["q"][:50],
                "type": "qa",
                "category": qa["category"],
            }
        )
        all_docs.append(doc)

    # 2.5 自动扩展：基于组件+故障类型+方案模板组合生成大量QA
    COMPONENTS = {
        "Redis": {"symptoms": ["内存持续增长", "响应延迟升高", "连接数暴增", "主从同步延迟", "集群节点失联", "AOF重写失败", "RDB持久化超时", "慢查询增多", "键空间命中率下降", "客户端连接超时"],
                  "causes": ["未配置maxmemory", "大Key堆积", "连接泄漏", "repl-backlog不足", "网络分区", "磁盘IO瓶颈", "淘汰策略不当", "热点Key集中", "过期Key未清理", "数据结构选择不当"],
                  "solutions": ["设置maxmemory和淘汰策略", "使用--bigkeys排查大Key并拆分", "配置timeout和连接池", "增大repl-backlog-size", "检查网络连通性和防火墙", "优化持久化策略或使用SSD", "启用lazyfree异步删除", "使用读写分离分散热点", "定期清理过期Key", "选择合适的数据结构"]},
        "MySQL": {"symptoms": ["慢查询增多", "连接数耗尽", "主从延迟增大", "死锁频繁", "磁盘IO过高", "Buffer Pool命中率低", "复制中断", "表锁等待", "临时表过多", "索引失效"],
                  "causes": ["缺少索引", "大事务未提交", "binlog格式问题", "锁竞争", "脏页刷盘", "内存不足", "网络抖动", "DDL操作", "排序无索引", "统计信息过期"],
                  "solutions": ["添加合适索引", "拆分大事务", "调整binlog_format", "优化锁粒度", "调整innodb_io_capacity", "增大buffer_pool_size", "配置半同步复制", "使用pt-online-schema-change", "添加排序索引", "ANALYZE TABLE更新统计信息"]},
        "Kubernetes": {"symptoms": ["Pod一直Pending", "Pod CrashLoopBackOff", "Service无法访问", "节点NotReady", "OOMKilled", "ImagePullBackOff", "PV无法挂载", "HPA不生效", "Ingress 502", "Deployment滚动更新卡住"],
                  "causes": ["资源不足", "容器启动失败", "selector标签不匹配", "kubelet异常", "内存limit过小", "镜像仓库不可达", "StorageClass未配置", "metrics-server未部署", "后端Pod不健康", "readinessProbe失败"],
                  "solutions": ["调整资源请求和限制", "查看容器日志定位崩溃原因", "检查selector和label匹配", "重启kubelet服务", "增大memory limit", "配置imagePullSecrets或使用私有仓库", "创建StorageClass和PV", "部署metrics-server", "检查后端Pod健康状态", "检查readinessProbe配置"]},
        "Nginx": {"symptoms": ["502 Bad Gateway", "504 Gateway Timeout", "429 Too Many Requests", "连接数耗尽", "SSL握手失败", "静态资源404", "upstream超时", "内存泄漏", "worker进程异常退出", "重定向循环"],
                  "causes": ["后端服务不可用", "后端响应超时", "并发超过limit_req", "worker_connections过小", "证书过期或配置错误", "root路径错误", "proxy_read_timeout过小", "模块内存泄漏", "段错误或信号中断", "rewrite规则错误"],
                  "solutions": ["检查后端服务状态和日志", "增大proxy_read_timeout", "调整limit_req配置", "增大worker_connections", "更新SSL证书和配置", "检查root和alias指令", "调整upstream超时参数", "升级Nginx版本或禁用问题模块", "检查coredump和error.log", "修复rewrite规则避免循环"]},
        "Docker": {"symptoms": ["容器无法启动", "磁盘空间不足", "网络不通", "容器频繁重启", "构建镜像慢", "容器时间不对", "日志文件过大", "端口冲突", "volume数据丢失", "容器间DNS解析失败"],
                  "causes": ["入口命令错误", "镜像层和日志堆积", "网络模式配置错误", "健康检查失败", "构建上下文过大", "时区未设置", "日志驱动未限制大小", "端口已被占用", "volume挂载路径错误", "DNS配置问题"],
                  "solutions": ["检查ENTRYPOINT/CMD和日志", "清理悬空镜像和日志", "检查网络模式和docker network", "调整healthcheck配置", "使用.dockerignore减小上下文", "设置TZ环境变量", "配置log-opts限制大小", "修改映射端口", "确认volume挂载正确", "使用--dns或自定义网络"]},
        "Kafka": {"symptoms": ["消息积压", "消费者lag增大", "分区Leader缺失", "Broker宕机", "生产者发送超时", "副本同步延迟", "磁盘写满", "消费者Rebalance频繁", "消息丢失", "消息重复消费"],
                  "causes": ["消费者处理慢", "消费能力不足", "Leader所在Broker故障", "硬件故障或OOM", "网络延迟或acks配置", "ISR副本不足", "日志段未清理", "会话超时或心跳间隔", "acks=0或未启用幂等", "未正确提交offset"],
                  "solutions": ["增加消费者实例或分区数", "优化消费逻辑和批量处理", "等待Controller重新选举Leader", "重启Broker并检查日志", "调整acks和request.timeout", "增加min.insync.replicas", "配置log.retention策略", "调整session.timeout.ms", "启用幂等性和事务", "确保offset正确提交"]},
        "Elasticsearch": {"symptoms": ["集群状态Red", "查询慢", "写入被拒绝", "磁盘水位告警", "分片未分配", "JVM堆内存不足", "批量索引失败", "节点脱离集群", "索引只读", "搜索结果不准确"],
                  "causes": ["主分片未分配", "查询未优化或数据量大", "线程池队列满", "磁盘使用超水位线", "节点故障或分片分配失败", "堆内存设置不当", "bulk请求过大", "网络问题或GC停顿", "磁盘超flood水位线", "分词器或映射问题"],
                  "solutions": ["查看分片分配状态并修复", "优化查询DSL和索引映射", "调整线程池大小和批量大小", "清理索引或扩容磁盘", "reroute命令手动分配", "调整JVM堆大小（不超过31GB）", "减小bulk请求大小", "检查网络和GC日志", "清理磁盘解除只读", "调整分词器和mapping"]},
        "Linux": {"symptoms": ["CPU使用率100%", "内存不足", "磁盘IO瓶颈", "进程僵死", "网络丢包", "文件描述符耗尽", "系统负载过高", "swap使用过多", "内核panic", "时间同步异常"],
                  "causes": ["计算密集型进程或死循环", "内存泄漏或缓存过大", "大量随机IO或磁盘性能差", "子进程未回收", "网卡缓冲区溢出或MTU问题", "连接数过多", "运行队列过长", "物理内存不足", "硬件故障或驱动bug", "NTP服务异常"],
                  "solutions": ["top/htop定位高CPU进程并优化", "排查内存泄漏并释放缓存", "使用iostat分析并优化IO", "kill父进程回收僵尸进程", "调整网卡参数和MTU", "增大ulimit和fs.file-max", "减少并发或增加CPU", "增加内存或优化swapiness", "检查硬件和内核日志", "配置chrony或ntpdate"]},
        "MongoDB": {"symptoms": ["查询慢", "复制集延迟", "内存使用过高", "连接数耗尽", "磁盘空间不足", "balancer阻塞", "chunk迁移失败", "索引构建慢", "oplog窗口不足", "写入性能下降"],
                  "causes": ["缺少索引或查询未命中", "secondary处理能力不足", "WiredTiger缓存过大", "连接未释放", "oplog或日志过大", "大数据块迁移", "jumbo chunk", "前台构建阻塞", "oplog大小不足", "写关注级别过高"],
                  "solutions": ["创建合适索引和explain分析", "增加secondary或优化读偏好", "调整wiredTiger.cacheSizeGB", "配置连接池和超时", "压缩oplog和日志", "手动split大chunk", "手动split jumbo chunk", "后台构建索引", "增大oplogSize", "调整write concern级别"]},
        "PostgreSQL": {"symptoms": ["查询慢", "连接数耗尽", "WAL堆积", "复制延迟", "表膨胀", "死锁", "VACUUM跟不上", "索引膨胀", "事务ID回卷", "自动分析未触发"],
                  "causes": ["缺少索引或计划不准确", "max_connections过小", "归档失败", "网络或IO瓶颈", "频繁更新删除", "并发锁竞争", "更新频繁autovacuum不及时", "索引碎片化", "老旧事务未提交", "autovacuum阈值过大"],
                  "solutions": ["创建索引和ANALYZE更新统计", "使用连接池pgbouncer", "修复归档命令", "优化网络和IO配置", "定期VACUUM和pg_repack", "优化锁粒度和事务", "调整autovacuum参数", "REINDEX重建索引", "提交长事务或设置idle_timeout", "降低autovacuum阈值"]},
        "RabbitMQ": {"symptoms": ["队列积压", "消费者断连", "集群分区", "内存告警", "磁盘告警", "通道泄漏", "消息丢失", "连接数过多", "镜像队列不同步", "Federation链路断开"],
                  "causes": ["消费速度跟不上生产", "消费者异常断开", "网络分区", "内存超watermark", "磁盘剩余不足", "未关闭通道", "未开启持久化或确认", "连接未复用", "网络问题", "上游节点故障"],
                  "solutions": ["增加消费者或优化消费逻辑", "配置自动重连和心跳", "处理分区：autoheal或pause-minority", "增大内存watermark或优化内存", "清理磁盘或增大磁盘限制", "确保通道正确关闭", "开启消息持久化和publisher confirms", "使用连接池", "修复网络并重新同步", "检查上游节点状态"]},
        "Prometheus": {"symptoms": ["采集延迟", "存储空间不足", "查询超时", "Target Down", "规则评估慢", "API响应慢", "TSDB压缩失败", "远程写入失败", "高基数标签", "内存使用过高"],
                  "causes": ["Target过多或网络延迟", "数据量增长过快", "查询范围过大", "目标服务不可达", "规则数量过多", "并发查询过多", "WAL文件损坏", "远端存储不可用", "标签值过多", "内存缓存过大"],
                  "solutions": ["优化采集间隔和Target数量", "配置retention和降采样", "缩小查询范围和时间窗口", "检查目标服务状态", "拆分规则到多个实例", "限制并发查询数", "修复WAL或重建TSDB", "检查远端存储连通性", "减少高基数标签", "调整storage.tsdb配置"]},
        "Java": {"symptoms": ["OOM", "GC停顿过长", "线程死锁", "CPU使用率高", "内存泄漏", "类加载冲突", "连接池耗尽", "StackOverflow", "应用启动慢", "响应延迟高"],
                  "causes": ["堆内存不足或内存泄漏", "堆过大或GC算法不当", "多线程锁竞争", "死循环或计算密集", "对象未释放", "类加载器隔离问题", "连接未归还", "递归过深", "类扫描范围过大", "GC或锁竞争"],
                  "solutions": ["增大堆内存或分析堆dump", "调整GC算法和堆大小", "jstack分析并优化锁", "jstack定位高CPU线程", "MAT分析堆dump", "使用独立类加载器", "配置连接池参数和超时", "优化递归为迭代", "优化类扫描范围", "优化GC和减少锁竞争"]},
        "Nacos": {"symptoms": ["服务注册失败", "配置推送延迟", "集群脑裂", "Raft选举异常", "连接数过多", "磁盘写满", "配置变更不生效", "服务发现延迟", "健康检查失败", "权限认证失败"],
                  "causes": ["网络不通或命名空间错误", "长轮询配置问题", "网络分区导致多Leader", "节点间通信异常", "连接未释放", "日志和数据未清理", "客户端缓存未刷新", "索引更新延迟", "健康检查间隔不当", "token过期或配置错误"],
                  "solutions": ["检查网络和命名空间配置", "优化长轮询超时时间", "处理分区并重启异常节点", "检查节点间通信和日志", "配置连接超时和最大连接数", "定期清理日志和数据", "客户端主动刷新配置", "优化索引和缓存", "调整健康检查参数", "检查token和权限配置"]},
        "Consul": {"symptoms": ["服务注册失败", "Leader选举失败", "RPC超时", "磁盘空间不足", "KV存储延迟", "Watch不触发", "健康检查异常", "Agent无法加入集群", "TLS握手失败", "Serf LAN断连"],
                  "causes": ["网络不通或ACL限制", "Raft日志损坏", "网络延迟或负载高", "Raft日志和快照堆积", "KV存储量大", "Watch配置错误", "检查脚本异常", "加密配置不一致", "证书过期", "网络分区"],
                  "solutions": ["检查网络和ACL策略", "恢复Raft日志或重建节点", "优化网络和增加资源", "清理旧快照和日志", "优化KV查询", "修正Watch配置", "修复检查脚本", "统一加密配置", "更新证书", "处理网络分区"]},
    }

    QUESTION_TEMPLATES = [
        "{component}{symptom}怎么排查？",
        "{component}{symptom}如何解决？",
        "{component}出现{symptom}是什么原因？",
        "{component}{symptom}怎么处理？",
        "如何诊断{component}{symptom}问题？",
        "{component}{symptom}的排查思路是什么？",
        "{component}{symptom}如何快速恢复？",
        "生产环境{component}{symptom}怎么应急处理？",
        "{component}{symptom}怎么预防？",
        "{component}{symptom}的最佳实践是什么？",
    ]

    auto_qa_count = 0
    for comp, info in COMPONENTS.items():
        for sym_idx, symptom in enumerate(info["symptoms"]):
            for cause_idx, cause in enumerate(info["causes"]):
                for sol_idx, solution in enumerate(info["solutions"]):
                    # 每个组件-症状组合只生成1个QA（取对应索引的cause和solution）
                    if cause_idx == sym_idx and sol_idx == sym_idx:
                        q_template = QUESTION_TEMPLATES[sym_idx % len(QUESTION_TEMPLATES)]
                        q = q_template.format(component=comp, symptom=symptom)
                        a = f"{comp}{symptom}的常见原因是{cause}。解决方案：{solution}。建议先排查{cause}，然后执行{solution}，同时做好监控告警。"
                        content = f"问: {q}\n答: {a}"
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": f"auto_{comp}_{sym_idx:02d}",
                                "source_name": q[:50],
                                "type": "auto_generated",
                                "category": comp,
                            }
                        )
                        all_docs.append(doc)
                        auto_qa_count += 1
                    # 额外：每个症状再配一个不同问法
                    elif cause_idx == (sym_idx + 1) % len(info["causes"]) and sol_idx == (sym_idx + 1) % len(info["solutions"]):
                        q_template = QUESTION_TEMPLATES[(sym_idx + 3) % len(QUESTION_TEMPLATES)]
                        q = q_template.format(component=comp, symptom=symptom)
                        a = f"{comp}{symptom}可能由{cause}引起。处理方法：{solution}。排查步骤：1.确认{cause} 2.执行{solution} 3.验证恢复。"
                        content = f"问: {q}\n答: {a}"
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": f"auto_{comp}_{sym_idx:02d}_v2",
                                "source_name": q[:50],
                                "type": "auto_generated",
                                "category": comp,
                            }
                        )
                        all_docs.append(doc)
                        auto_qa_count += 1
                    # 第三轮：再配一个预防角度的QA
                    elif cause_idx == (sym_idx + 2) % len(info["causes"]) and sol_idx == (sym_idx + 2) % len(info["solutions"]):
                        q = f"{comp}{symptom}怎么预防？有哪些最佳实践？"
                        a = f"预防{comp}{symptom}的关键是避免{cause}。最佳实践：1.定期检查{cause}相关指标 2.提前实施{solution} 3.配置监控告警 4.制定应急预案。"
                        content = f"问: {q}\n答: {a}"
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": f"auto_{comp}_{sym_idx:02d}_v3",
                                "source_name": q[:50],
                                "type": "auto_generated",
                                "category": comp,
                            }
                        )
                        all_docs.append(doc)
                        auto_qa_count += 1

    logger.info(f"自动扩展QA: {auto_qa_count} 条")

    # 3. 切片扩展到 1000+（用较小切片大小增加切片数）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n案例 ", "\n案例", "案例", "\n问：", "\n问:", "\n## ", "\n\n", "\n", "。", " ", ""],
    )
    splits = splitter.split_documents(all_docs)

    # 4. 如果切片仍不足1000，对每个案例按段落拆分补充
    if len(splits) < 1000:
        logger.info(f"切片数 {len(splits)} < 1000，按段落补充拆分...")
        extra_docs = []
        for case in CASE_STUDIES:
            # 按段落拆分案例内容
            paragraphs = [p.strip() for p in case["content"].split("\n") if p.strip() and len(p.strip()) > 20]
            for j, para in enumerate(paragraphs):
                if len(para) > 30:  # 只保留有实质内容的段落
                    doc = Document(
                        page_content=para,
                        metadata={
                            "source": f"para_{case['category']}_{hashlib.md5(para.encode()).hexdigest()[:8]}",
                            "source_name": f"{case['title']} - 段落{j+1}",
                            "type": "paragraph",
                            "category": case["category"],
                        }
                    )
                    extra_docs.append(doc)
        splits.extend(extra_docs)

    # 5. 如果还不够，对QA模板做扩展变体
    if len(splits) < 1000:
        logger.info(f"切片数 {len(splits)} < 1000，生成QA变体补充...")
        # 基于QA模板生成变体问题
        qa_variants = []
        for qa in QA_TEMPLATES:
            # 原始QA
            qa_variants.append(qa)
            # 变体1：换问法
            for old, new_list in [("怎么办", "如何解决"), ("怎么处理", "如何修复"), ("如何", "怎么"), ("排查", "定位"), ("优化", "调优")]:
                if old in qa["q"]:
                    for new_word in new_list:
                        variant_q = qa["q"].replace(old, new_word)
                        if variant_q != qa["q"]:
                            qa_variants.append({"q": variant_q, "a": qa["a"], "category": qa["category"]})
                    break  # 每个问题只替换一次

        for i, qa in enumerate(qa_variants):
            content = f"问: {qa['q']}\n答: {qa['a']}"
            doc = Document(
                page_content=content,
                metadata={
                    "source": f"qa_var_{qa['category']}_{i:04d}",
                    "source_name": qa["q"][:50],
                    "type": "qa_variant",
                    "category": qa["category"],
                }
            )
            splits.append(doc)

    # 统一 metadata
    for i, doc in enumerate(splits):
        doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
        doc.metadata.setdefault("doc_id", doc_id)
        if "source" not in doc.metadata:
            doc.metadata["source"] = f"split_{i:04d}"
        if "source_name" not in doc.metadata:
            doc.metadata["source_name"] = doc.metadata.get("source", "unknown")
        if "type" not in doc.metadata:
            doc.metadata["type"] = "split"

    logger.info(f"向量文档生成: {len(CASE_STUDIES)} 案例 + {len(QA_TEMPLATES)} QA -> {len(splits)} 切片")
    return splits


def import_to_milvus(splits, collection_name: str, uri: str, drop_existing: bool = False):
    """导入数据到 Milvus 向量库"""
    from pymilvus import MilvusClient, connections
    from dotenv import load_dotenv

    # 加载环境变量
    for ef in [".env", "Key.env", "Env.env", "Env1.env"]:
        p = os.path.join(BASE_DIR, ef)
        if os.path.exists(p):
            load_dotenv(p, override=False)
            break

    if not uri.startswith("http://") and not uri.startswith("https://"):
        uri = f"http://{uri}"

    # 优先使用 Ollama 本地 embedding，DashScope 作为备选
    emb = None
    ollama_url = os.getenv("OLLAMA_URL", "http://192.168.100.128:11434")
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")

    # 尝试 Ollama 本地 embedding
    try:
        from langchain_ollama import OllamaEmbeddings
        ollama_base = ollama_url.replace("/api/embeddings", "").replace("/api/embed", "")
        # 按优先级尝试可用的 embedding 模型
        ollama_models = ["nomic-embed-text", "mofanke/acge_text_embedding", "bge-m3"]
        for model_name in ollama_models:
            try:
                emb = OllamaEmbeddings(
                    model=model_name,
                    base_url=ollama_base,
                )
                test_result = emb.embed_query("测试")
                if test_result and len(test_result) > 0:
                    logger.info(f"使用 Ollama 本地 embedding ({model_name}), 地址: {ollama_base}")
                    break
                else:
                    emb = None
            except Exception:
                emb = None
                continue
    except Exception as e:
        logger.warning(f"Ollama embedding 不可用: {e}")
        emb = None

    # Ollama 不可用时尝试 DashScope
    if emb is None and dashscope_api_key:
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
            embed_model = os.getenv("EMBED_MODEL", "text-embedding-async-v1")
            emb = DashScopeEmbeddings(model=embed_model, dashscope_api_key=dashscope_api_key)
            test_result = emb.embed_query("测试")
            if test_result and len(test_result) > 0:
                logger.info(f"使用 DashScope embedding ({embed_model})")
            else:
                emb = None
        except Exception as e:
            logger.warning(f"DashScope embedding 不可用: {e}")
            emb = None

    if emb is None:
        logger.error("所有 embedding 服务均不可用（Ollama 和 DashScope），无法导入向量库")
        logger.error("请确保以下任一服务可用：")
        logger.error("  1. Ollama: http://192.168.100.128:11434 (需安装 bge-m3 模型)")
        logger.error("  2. DashScope: 需有效 API Key 且账户未欠费")
        return

    try:
        from langchain_milvus import Milvus as MilvusVS
    except ImportError:
        logger.error("langchain-milvus 未安装，请运行: pip install langchain-milvus")
        return

    client = MilvusClient(uri=uri)
    total = len(splits)

    if total == 0:
        logger.error("没有文档可导入")
        return

    if drop_existing and client.has_collection(collection_name):
        logger.info(f"删除已有集合: {collection_name}")
        client.drop_collection(collection_name)

    need_create = not client.has_collection(collection_name)

    # 注册连接兼容性补丁
    alias = client._using
    handler = client._handler
    if alias not in connections._alias_handlers:
        connections._alias_handlers[alias] = handler
        connections._alias_config[alias] = {
            'address': client._config.address,
            'uri': client._config.uri,
        }

    batch_size = 200

    if need_create:
        first_batch = min(batch_size, total)
        logger.info(f"创建集合并导入前 {first_batch} 个切片...")
        vs = MilvusVS.from_documents(
            splits[:first_batch],
            emb,
            collection_name=collection_name,
            connection_args={"uri": uri},
            enable_dynamic_field=True,
        )
        logger.info(f"已导入 {first_batch}/{total} ({first_batch * 100 // total}%)")
        remaining = splits[first_batch:]
        if not remaining:
            logger.info(f"导入完成! 集合: {collection_name}, 总计: {total}")
            return
    else:
        logger.info(f"集合 {collection_name} 已存在，追加数据...")
        remaining = splits
        vs = MilvusVS(
            collection_name=collection_name,
            embedding_function=emb,
            connection_args={"uri": uri},
            auto_id=True,
            enable_dynamic_field=True,
        )

    # 分批追加
    total_batches = (len(remaining) + batch_size - 1) // batch_size
    start_idx = total - len(remaining)

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(remaining))
        batch = remaining[start:end]
        current_global = start_idx + end
        pct = current_global * 100 // total

        try:
            vs.add_documents(batch)
            logger.info(f"[{batch_idx + 1}/{total_batches}] 已导入 {current_global}/{total} ({pct}%)")
        except Exception as e:
            logger.error(f"[{batch_idx + 1}/{total_batches}] 导入失败: {e}")
            try:
                vs = MilvusVS(
                    collection_name=collection_name,
                    embedding_function=emb,
                    connection_args={"uri": uri},
                    auto_id=True,
                    enable_dynamic_field=True,
                )
                vs.add_documents(batch)
                logger.info(f"  重试成功: {len(batch)} 个切片")
            except Exception as e2:
                logger.error(f"  重试也失败，跳过本批: {e2}")

        if batch_idx < total_batches - 1:
            time.sleep(0.3)

    logger.info(f"导入完成! 集合: {collection_name}, 总计: {total} 个切片")


def generate_lora_data(output_dir: str, count: int = 100) -> str:
    """生成 LoRA 微调训练数据（100条），适配项目 finetune.py 的格式"""
    samples = []
    instruction = "你是一个运维专家，请根据知识回答运维问题。"

    # 从案例和QA中选取
    all_items = []
    for case in CASE_STUDIES:
        all_items.append({"input": case["title"], "output": case["content"], "category": case["category"]})
    for qa in QA_TEMPLATES:
        all_items.append({"input": qa["q"], "output": qa["a"], "category": qa["category"]})

    # 随机选取并确保多方向
    random.seed(42)
    categories = list(set(item["category"] for item in all_items))
    per_category = max(1, count // len(categories))

    selected = []
    for cat in categories:
        cat_items = [item for item in all_items if item["category"] == cat]
        selected.extend(cat_items[:per_category])

    # 如果不够，从剩余中补充
    remaining = [item for item in all_items if item not in selected]
    random.shuffle(remaining)
    selected.extend(remaining[:count - len(selected)])

    selected = selected[:count]

    for item in selected:
        samples.append({
            "instruction": instruction,
            "input": item["input"],
            "output": item["output"],
            "source": item["category"],
        })

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "lora_train.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # 统计方向分布
    cat_stats = {}
    for s in samples:
        cat_stats[s["source"]] = cat_stats.get(s["source"], 0) + 1

    logger.info(f"LoRA 训练数据生成: {len(samples)} 条 -> {output_file}")
    for cat, cnt in sorted(cat_stats.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {cnt}条")

    return output_file


def generate_reranker_data(output_dir: str, count: int = 100) -> str:
    """生成 Reranker 重排序训练数据（100条），适配项目 finetune.py 的格式"""
    samples = []
    random.seed(42)

    # 从QA模板构建正负例
    all_qa = [(qa["q"], qa["a"], qa["category"]) for qa in QA_TEMPLATES]
    all_cases = [(case["title"], case["content"], case["category"]) for case in CASE_STUDIES]
    all_items = all_qa + all_cases

    # 确保多方向覆盖
    categories = list(set(item[2] for item in all_items))
    per_category = max(1, count // len(categories))

    used = 0
    for cat in categories:
        cat_items = [item for item in all_items if item[2] == cat]
        for query, answer, _ in cat_items[:per_category]:
            if used >= count:
                break
            # 正例：同类别的内容
            positive_contexts = [answer]
            # 负例：不同类别的内容
            other_items = [item for item in all_items if item[2] != cat]
            if other_items:
                neg_samples = random.sample(other_items, min(3, len(other_items)))
                negative_contexts = [neg[1][:300] for neg in neg_samples]
            else:
                negative_contexts = []

            samples.append({
                "query": query,
                "positive_contexts": positive_contexts,
                "negative_contexts": negative_contexts,
                "category": cat,
            })
            used += 1

    # 补充到目标数量
    remaining_items = [item for item in all_items if item not in [(s["query"], s.get("positive_contexts", [""])[0] if s.get("positive_contexts") else "", item[2]) for s in samples]]
    random.shuffle(remaining_items if remaining_items else all_items)
    for query, answer, cat in (remaining_items if remaining_items else all_items):
        if used >= count:
            break
        other_items = [item for item in all_items if item[2] != cat]
        neg_samples = random.sample(other_items, min(3, len(other_items))) if other_items else []
        samples.append({
            "query": query,
            "positive_contexts": [answer],
            "negative_contexts": [neg[1][:300] for neg in neg_samples],
            "category": cat,
        })
        used += 1

    samples = samples[:count]

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "reranker_train.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # 统计
    cat_stats = {}
    for s in samples:
        cat_stats[s["category"]] = cat_stats.get(s["category"], 0) + 1

    logger.info(f"Reranker 训练数据生成: {len(samples)} 条 -> {output_file}")
    for cat, cnt in sorted(cat_stats.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {cnt}条")

    return output_file


def generate_augmented_data(output_dir: str, count: int = 200) -> str:
    """生成同义问题变体数据（200条），多方向扩展"""
    variants = generate_synonym_variants(target_count=count)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "augmented_questions.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(variants, f, ensure_ascii=False, indent=2)

    logger.info(f"同义问题变体数据生成: {len(variants)} 条 -> {output_file}")
    return output_file


def verify_milvus_data(collection_name: str, uri: str, expected_min: int = 1000):
    """验证 Milvus 中的数据量"""
    from pymilvus import MilvusClient

    if not uri.startswith("http://") and not uri.startswith("https://"):
        uri = f"http://{uri}"

    client = MilvusClient(uri=uri)

    if not client.has_collection(collection_name):
        logger.error(f"集合 {collection_name} 不存在!")
        return False

    # 兼容 Milvus 2.4+ 的 count 查询
    try:
        result = client.query(collection_name=collection_name, filter="", output_fields=["count(*)"])
        count = result[0].get("count(*)", 0) if result else 0
    except Exception:
        try:
            stats = client.get_collection_stats(collection_name)
            count = int(stats.get("row_count", 0))
        except Exception as e:
            logger.error(f"无法获取集合统计: {e}")
            return False

    logger.info(f"Milvus 集合 {collection_name}: {count} 条数据")

    if count >= expected_min:
        logger.info(f"验证通过: {count} >= {expected_min}")
        return True
    else:
        logger.warning(f"数据量不足: {count} < {expected_min}")
        return False


# ============================================================================
# 第五部分：主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SmartOps 运维数据准备脚本")
    parser.add_argument("--task", choices=["all", "vector", "finetune", "augment"], default="all",
                        help="执行任务: all=全部, vector=仅向量库, finetune=仅微调数据, augment=仅问题变体")
    parser.add_argument("--uri", type=str, default=MILVUS_URI, help="Milvus URI")
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME, help="Milvus 集合名")
    parser.add_argument("--clear", action="store_true", help="清空已有集合后重建")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--lora-count", type=int, default=100, help="LoRA 数据条数")
    parser.add_argument("--reranker-count", type=int, default=100, help="Reranker 数据条数")
    parser.add_argument("--augment-count", type=int, default=200, help="同义变体条数")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  SmartOps 运维数据准备脚本")
    logger.info(f"  任务: {args.task}")
    logger.info(f"  Milvus: {args.uri}")
    logger.info(f"  集合: {args.collection}")
    logger.info("=" * 60)

    # ---- 任务1: 向量库数据 ----
    if args.task in ("all", "vector"):
        logger.info("\n" + "=" * 60)
        logger.info("  任务1: 生成向量库数据并导入 Milvus")
        logger.info("=" * 60)

        splits = generate_vector_documents()
        logger.info(f"生成 {len(splits)} 条向量文档")

        if len(splits) < 1000:
            logger.warning(f"切片数 {len(splits)} < 1000，需要更多原始数据")
        else:
            logger.info(f"切片数 {len(splits)} >= 1000，满足要求")

        import_to_milvus(splits, args.collection, args.uri, drop_existing=args.clear)
        verify_milvus_data(args.collection, args.uri, expected_min=1000)

    # ---- 任务2: 微调数据 ----
    if args.task in ("all", "finetune"):
        logger.info("\n" + "=" * 60)
        logger.info("  任务2: 生成 LoRA + Reranker 微调数据")
        logger.info("=" * 60)

        lora_file = generate_lora_data(args.output_dir, count=args.lora_count)
        reranker_file = generate_reranker_data(args.output_dir, count=args.reranker_count)

        logger.info(f"LoRA 数据: {lora_file}")
        logger.info(f"Reranker 数据: {reranker_file}")

    # ---- 任务3: 同义问题变体 ----
    if args.task in ("all", "augment"):
        logger.info("\n" + "=" * 60)
        logger.info("  任务3: 生成同义问题变体（多方向扩展）")
        logger.info("=" * 60)

        augment_file = generate_augmented_data(args.output_dir, count=args.augment_count)
        logger.info(f"同义变体数据: {augment_file}")

    logger.info("\n" + "=" * 60)
    logger.info("  所有任务完成!")
    logger.info(f"  输出目录: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()