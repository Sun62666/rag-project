import os
import re
import json
import time
import logging
import argparse
import hashlib
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "crawled")
STATE_FILE = os.path.join(BASE_DIR, "data", "crawler_state.json")

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp",
    ".mp4", ".mp3", ".avi", ".mov", ".wmv", ".flv",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".pdf", ".doc", ".docx", ".ppt", ".pptx",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
    ".rss", ".atom", ".json",
}

SKIP_URL_PATTERNS = [
    "/api/", "/feed/", "/search?", "/login", "/signup",
    "/privacy", "/terms", "/legal", "/cookie",
    "#", "javascript:", "mailto:",
]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class WebCrawler:

    def __init__(self, output_dir: str = DATA_DIR, delay: float = 1.5, max_pages: int = 100):
        self.output_dir = output_dir
        self.delay = delay
        self.max_pages = max_pages
        self.visited: Set[str] = set()
        self.session = None
        self._save_count = 0
        _ensure_dir(output_dir)

    def _get_session(self):
        if self.session is None:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            self.session = requests.Session()
            self.session.headers.update(BROWSER_HEADERS)

            retry_strategy = Retry(
                total=3,
                backoff_factor=1.0,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "HEAD"],
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)

        return self.session

    def _fetch(self, url: str, timeout: int = 30) -> Optional[str]:
        session = self._get_session()
        for attempt in range(3):
            try:
                resp = session.get(url, timeout=timeout, allow_redirects=True)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 5))
                    logger.warning(f"限流 {url}, 等待 {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.text
            except Exception as e:
                if attempt < 2:
                    wait = (attempt + 1) * 2
                    logger.warning(f"请求失败 (重试 {attempt + 1}/3): {url} - {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"请求失败 {url}: {e}")
                    return None

    def _is_content_url(self, url: str) -> bool:
        parsed = urlparse(url)
        path_lower = parsed.path.lower()

        for ext in SKIP_EXTENSIONS:
            if path_lower.endswith(ext):
                return False

        for pattern in SKIP_URL_PATTERNS:
            if pattern in url.lower():
                return False

        return True

    def _is_html_content(self, url: str, html: str) -> bool:
        if not html or len(html) < 100:
            return False

        stripped = html.strip().lower()
        if stripped.startswith("<?xml") or stripped.startswith("<sitemapindex") or stripped.startswith("<urlset"):
            return False

        html_markers = ["<html", "<!doctype html", "<head", "<body", "<div", "<article", "<main"]
        return any(m in stripped[:2000].lower() for m in html_markers)

    def _parse_sitemap(self, xml_content: str, base_url: str = "") -> tuple:
        page_urls = []
        sitemap_urls = []

        if "<sitemapindex" in xml_content:
            for match in re.finditer(r'<sitemap>.*?<loc>(.*?)</loc>.*?</sitemap>', xml_content, re.DOTALL):
                sitemap_urls.append(match.group(1).strip())
            logger.info(f"发现 sitemap 索引，包含 {len(sitemap_urls)} 个子 sitemap")

        elif "<urlset" in xml_content:
            for match in re.finditer(r'<url>.*?<loc>(.*?)</loc>.*?</url>', xml_content, re.DOTALL):
                url = match.group(1).strip()
                if self._is_content_url(url):
                    page_urls.append(url)
            logger.info(f"发现 urlset，包含 {len(page_urls)} 个页面 URL")

        else:
            for match in re.finditer(r'<loc>(.*?)</loc>', xml_content):
                url = match.group(1).strip()
                if url.endswith(".xml"):
                    sitemap_urls.append(url)
                elif self._is_content_url(url):
                    page_urls.append(url)

        return page_urls, sitemap_urls

    def _html_to_markdown(self, html: str, base_url: str = "") -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return self._simple_html_extract(html)

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
            tag.decompose()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/") or href.startswith("./"):
                a["href"] = urljoin(base_url, href)

        lines = []
        for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            level = int(el.name[1])
            text = el.get_text(strip=True)
            if text:
                lines.append(f"\n{'#' * level} {text}\n")

        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if text:
                lines.append(f"\n{text}\n")

        for pre in soup.find_all("pre"):
            code = pre.get_text()
            lang = ""
            code_tag = pre.find("code")
            if code_tag and code_tag.get("class"):
                for c in code_tag["class"]:
                    if c.startswith("language-") or c.startswith("lang-"):
                        lang = c.split("-", 1)[1]
                        break
            lines.append(f"\n```{lang}\n{code}\n```\n")

        for ul in soup.find_all("ul"):
            for li in ul.find_all("li"):
                lines.append(f"- {li.get_text(strip=True)}")
            lines.append("")

        for ol in soup.find_all("ol"):
            for i, li in enumerate(ol.find_all("li"), 1):
                lines.append(f"{i}. {li.get_text(strip=True)}")
            lines.append("")

        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for i, row in enumerate(rows):
                cells = row.find_all(["th", "td"])
                line = "| " + " | ".join(c.get_text(strip=True) for c in cells) + " |"
                lines.append(line)
                if i == 0:
                    lines.append("| " + " | ".join("---" for _ in cells) + " |")
            lines.append("")

        content = "\n".join(lines)
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip()

    def _simple_html_extract(self, html: str) -> str:
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL)
        text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL)
        text = re.sub(r'<h([1-6])[^>]*>(.*?)</h\1>', r'\n\#\1 \2\n', text)
        text = re.sub(r'<p[^>]*>(.*?)</p>', r'\n\1\n', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _save_page(self, url: str, content: str, source_name: str) -> str:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        if path_parts:
            safe_name = re.sub(r'[^\w\-.]', '_', path_parts[-1])
            filename = f"{safe_name}_{url_hash}.md"
        else:
            filename = f"index_{url_hash}.md"

        save_dir = os.path.join(self.output_dir, source_name)
        _ensure_dir(save_dir)
        filepath = os.path.join(save_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# 来源: {url}\n")
            f.write(f"# 爬取时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(content)

        self._save_count += 1
        logger.info(f"已保存 [{self._save_count}]: {filename} ({len(content)} 字符)")
        return filepath

    def crawl_page(self, url: str, source_name: str = "misc") -> Optional[str]:
        if url in self.visited:
            return None
        if self._save_count >= self.max_pages:
            return None

        self.visited.add(url)

        if not self._is_content_url(url):
            logger.debug(f"跳过非内容 URL: {url}")
            return None

        logger.info(f"正在爬取: {url}")

        html = self._fetch(url)
        if not html:
            return None

        if not self._is_html_content(url, html):
            logger.debug(f"非 HTML 内容，跳过: {url}")
            return None

        markdown = self._html_to_markdown(html, base_url=url)
        if len(markdown) < 80:
            logger.warning(f"内容过短 ({len(markdown)} 字符)，跳过: {url}")
            return None

        filepath = self._save_page(url, markdown, source_name)
        time.sleep(self.delay)
        return filepath

    def crawl_sitemap(self, sitemap_url: str, source_name: str = "", max_pages: int = 0) -> List[str]:
        if not source_name:
            parsed = urlparse(sitemap_url)
            source_name = parsed.netloc.replace(".", "_")

        effective_max = max_pages or self.max_pages
        saved_files = []
        all_page_urls: List[str] = []
        visited_sitemaps: Set[str] = {sitemap_url}

        logger.info(f"正在获取 sitemap: {sitemap_url}")
        xml = self._fetch(sitemap_url)
        if not xml:
            logger.error(f"无法获取 sitemap: {sitemap_url}")
            return []

        page_urls, sitemap_urls = self._parse_sitemap(xml, sitemap_url)
        all_page_urls.extend(page_urls)

        max_sitemap_depth = 3
        current_sitemaps = sitemap_urls
        depth = 0
        while current_sitemaps and depth < max_sitemap_depth:
            depth += 1
            next_sitemaps = []
            for sm_url in current_sitemaps:
                if sm_url in visited_sitemaps:
                    continue
                visited_sitemaps.add(sm_url)

                logger.info(f"解析子 sitemap ({depth}): {sm_url}")
                sm_xml = self._fetch(sm_url)
                if not sm_xml:
                    continue

                sub_pages, sub_sitemaps = self._parse_sitemap(sm_xml, sm_url)
                all_page_urls.extend(sub_pages)
                next_sitemaps.extend(sub_sitemaps)
                time.sleep(self.delay)

            current_sitemaps = next_sitemaps

        all_page_urls = list(dict.fromkeys(all_page_urls))
        logger.info(f"共发现 {len(all_page_urls)} 个页面 URL，开始爬取 (最多 {effective_max} 页)")

        for i, url in enumerate(all_page_urls):
            if self._save_count >= effective_max:
                logger.info(f"已达到最大页数 {effective_max}")
                break

            result = self.crawl_page(url, source_name)
            if result:
                saved_files.append(result)

            if (i + 1) % 20 == 0:
                logger.info(f"进度: {i + 1}/{min(len(all_page_urls), effective_max)}")

        logger.info(f"sitemap 爬取完成: {len(saved_files)} 个页面")
        return saved_files

    def crawl_recursive(self, start_url: str, source_name: str = "", max_depth: int = 2, max_pages: int = 0, url_pattern: str = "") -> List[str]:
        if not source_name:
            parsed = urlparse(start_url)
            source_name = parsed.netloc.replace(".", "_")

        effective_max = max_pages or self.max_pages
        saved_files = []
        queue = [(start_url, 0)]
        base_domain = urlparse(start_url).netloc

        while queue and self._save_count < effective_max:
            url, depth = queue.pop(0)
            if depth > max_depth or url in self.visited:
                continue

            result = self.crawl_page(url, source_name)
            if result:
                saved_files.append(result)

            if depth < max_depth:
                html = self._fetch(url)
                if html and self._is_html_content(url, html):
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html, "html.parser")
                        for a in soup.find_all("a", href=True):
                            href = urljoin(url, a["href"])
                            href_parsed = urlparse(href)
                            if href_parsed.netloc != base_domain:
                                continue
                            if href_parsed.scheme not in ("http", "https"):
                                continue
                            if url_pattern and url_pattern not in href:
                                continue
                            if not self._is_content_url(href):
                                continue
                            if href not in self.visited and href not in [q[0] for q in queue]:
                                queue.append((href, depth + 1))
                    except ImportError:
                        pass

        logger.info(f"递归爬取完成: {len(saved_files)} 个页面")
        return saved_files


PRESET_SOURCES = {
    "k8s": {
        "name": "kubernetes",
        "sitemap": "https://kubernetes.io/zh-cn/sitemap.xml",
        "pattern": "/zh-cn/",
    },
    "redis": {
        "name": "redis",
        "url": "https://redis.com.cn/documentation.html",
        "pattern": "",
    },
    "nginx": {
        "name": "nginx",
        "url": "https://nginx.org/en/docs/",
        "pattern": "/docs/",
    },
    "mysql": {
        "name": "mysql",
        "url": "https://www.runoob.com/mysql/mysql-tutorial.html",
        "pattern": "/mysql/",
    },
    "docker": {
        "name": "docker",
        "sitemap": "https://docs.docker.com/sitemap.xml",
        "pattern": "/zh/",
    },
    "prometheus": {
        "name": "prometheus",
        "sitemap": "https://prometheus.io/sitemap.xml",
        "pattern": "/docs/",
    },
    "linux": {
        "name": "linux",
        "url": "https://man7.org/linux/man-pages/",
        "pattern": "",
    },
    "elasticsearch": {
        "name": "elasticsearch",
        "url": "https://www.elastic.co/guide/cn/elasticsearch/guide/current/index.html",
        "pattern": "/cn/elasticsearch/",
    },
    "rabbitmq": {
        "name": "rabbitmq",
        "sitemap": "https://www.rabbitmq.com/sitemap.xml",
        "pattern": "/docs/",
    },
    "mongodb": {
        "name": "mongodb",
        "url": "https://www.mongodb.com/docs/",
        "pattern": "/docs/",
    },
}


def crawl_batch(source_keys: List[str], max_pages: int = 50, delay: float = 1.5) -> Dict[str, int]:
    results = {}
    total = len(source_keys)

    for idx, key in enumerate(source_keys, 1):
        if key not in PRESET_SOURCES:
            logger.error(f"[{idx}/{total}] 未知预设源: {key}")
            results[key] = 0
            continue

        source = PRESET_SOURCES[key]
        logger.info(f"\n{'='*60}")
        logger.info(f"[{idx}/{total}] 开始爬取: {key} ({source['name']})")
        logger.info(f"{'='*60}")

        files = crawl_preset(key, max_pages=max_pages, delay=delay)
        results[key] = len(files)

        if idx < total:
            pause = 3
            logger.info(f"等待 {pause}s 后继续下一个源...")
            time.sleep(pause)

    return results


def crawl_preset(source_key: str, max_pages: int = 50, delay: float = 1.5) -> List[str]:
    if source_key not in PRESET_SOURCES:
        logger.error(f"未知预设源: {source_key}, 可选: {list(PRESET_SOURCES.keys())}")
        return []

    source = PRESET_SOURCES[source_key]
    crawler = WebCrawler(delay=delay, max_pages=max_pages)

    if source.get("sitemap"):
        return crawler.crawl_sitemap(
            source["sitemap"],
            source_name=source["name"],
        )
    elif source.get("url"):
        return crawler.crawl_recursive(
            source["url"],
            source_name=source["name"],
            url_pattern=source.get("pattern", ""),
        )
    else:
        logger.error(f"预设源 {source_key} 缺少 sitemap 或 url 配置")
        return []


def crawl_custom(url: str, source_name: str = "", max_depth: int = 2, max_pages: int = 50, delay: float = 1.5, url_pattern: str = "") -> List[str]:
    crawler = WebCrawler(delay=delay, max_pages=max_pages)

    if url.endswith(".xml") and "sitemap" in url.lower():
        return crawler.crawl_sitemap(url, source_name=source_name)
    else:
        return crawler.crawl_recursive(
            url,
            source_name=source_name,
            max_depth=max_depth,
            url_pattern=url_pattern,
        )


def main():
    parser = argparse.ArgumentParser(description="SmartOps 运维文档爬虫")
    parser.add_argument("action", choices=["preset", "batch", "custom", "list", "stats"], help="操作类型: preset=单个源, batch=批量爬取, custom=自定义URL")
    parser.add_argument("--source", type=str, help="预设源名称 (k8s/redis/nginx/mysql/docker/prometheus/linux/elasticsearch/rabbitmq/mongodb)")
    parser.add_argument("--sources", type=str, help="批量爬取: 逗号分隔的源名称，如 k8s,redis,docker 或 all")
    parser.add_argument("--url", type=str, help="自定义爬取 URL")
    parser.add_argument("--name", type=str, default="", help="自定义源名称")
    parser.add_argument("--max-pages", type=int, default=50, help="每个源最大爬取页数")
    parser.add_argument("--max-depth", type=int, default=2, help="递归深度")
    parser.add_argument("--delay", type=float, default=1.5, help="请求间隔(秒)")
    parser.add_argument("--pattern", type=str, default="", help="URL 过滤模式")
    parser.add_argument("--output", type=str, default=DATA_DIR, help="输出目录")

    args = parser.parse_args()

    if args.action == "list":
        print("可用的预设源 (均为中文文档):")
        print(f"  {'关键字':15s} {'名称':15s} {'地址'}")
        print(f"  {'-'*15} {'-'*15} {'-'*50}")
        for key, info in PRESET_SOURCES.items():
            method = info.get("sitemap", info.get("url", "未知"))
            print(f"  {key:15s} {info['name']:15s} {method}")
        print(f"\n批量爬取示例:")
        print(f"  python scripts/crawler.py batch --sources all --max-pages 30")
        print(f"  python scripts/crawler.py batch --sources k8s,redis,docker --max-pages 20")
        return

    if args.action == "stats":
        if not os.path.isdir(DATA_DIR):
            print("尚未爬取任何数据")
            return
        total_files = 0
        total_size = 0
        per_source = {}
        for root, dirs, files in os.walk(DATA_DIR):
            for f in files:
                if f.endswith(".md"):
                    total_files += 1
                    fpath = os.path.join(root, f)
                    total_size += os.path.getsize(fpath)
                    source_name = os.path.basename(root)
                    per_source[source_name] = per_source.get(source_name, 0) + 1
        print(f"已爬取: {total_files} 个文件, 总大小: {total_size / 1024:.1f} KB")
        if per_source:
            print(f"\n各源文件数:")
            for name, count in sorted(per_source.items(), key=lambda x: -x[1]):
                print(f"  {name:20s} {count} 个文件")
        return

    if args.action == "batch":
        sources_str = args.sources or "all"
        if sources_str.lower() == "all":
            source_keys = list(PRESET_SOURCES.keys())
        else:
            source_keys = [s.strip() for s in sources_str.split(",") if s.strip()]

        if not source_keys:
            print("请指定 --sources, 如: --sources all 或 --sources k8s,redis,docker")
            return

        print(f"即将批量爬取 {len(source_keys)} 个源: {', '.join(source_keys)}")
        print(f"每个源最多 {args.max_pages} 页, 间隔 {args.delay}s\n")

        results = crawl_batch(source_keys, max_pages=args.max_pages, delay=args.delay)

        print(f"\n{'='*60}")
        print(f"  批量爬取汇总")
        print(f"{'='*60}")
        total_files = 0
        for key, count in results.items():
            status = "OK" if count > 0 else "FAIL"
            print(f"  {status} {key:15s} {count} 个文件")
            total_files += count
        print(f"{'='*60}")
        print(f"  总计: {total_files} 个文件")
        print(f"\n导入知识库命令:")
        print(f"  python scripts/import_data.py import --path {DATA_DIR}")

    elif args.action == "preset":
        if not args.source:
            print("请指定 --source, 可选: " + ", ".join(PRESET_SOURCES.keys()))
            return
        files = crawl_preset(args.source, max_pages=args.max_pages, delay=args.delay)
        print(f"\n爬取完成! 共 {len(files)} 个文件")
        if files:
            source_name = PRESET_SOURCES[args.source]["name"]
            print(f"可以运行以下命令导入到知识库:")
            print(f"  python scripts/import_data.py import --path {os.path.join(DATA_DIR, source_name)}")

    elif args.action == "custom":
        if not args.url:
            print("请指定 --url")
            return
        files = crawl_custom(
            args.url,
            source_name=args.name,
            max_depth=args.max_depth,
            max_pages=args.max_pages,
            delay=args.delay,
            url_pattern=args.pattern,
        )
        print(f"\n爬取完成! 共 {len(files)} 个文件")


if __name__ == "__main__":
    main()
