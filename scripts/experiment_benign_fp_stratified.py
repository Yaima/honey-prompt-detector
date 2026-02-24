#!/usr/bin/env python3
"""
Benign FP Stratified Corpus Study
===================================

Tests Stage 3 token matching and Stage 1 heuristics on a benign-only corpus
stratified by category:
  - English prose
  - Spanish text
  - Chinese text
  - Arabic text
  - Code blocks (Python, JS)
  - JSON/structured data
  - Log lines
  - URLs / web content

Reports:
  - Stage 3 FPR (token match) per category — should be 0
  - Stage 1 heuristic FPR per category — expected >0 for some categories

Usage:
  cd honey-prompt-detector
  python scripts/experiment_benign_fp_stratified.py
"""

import secrets
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.honey_prompt_detector.core.detector import Detector
from src.honey_prompt_detector.core.heuristic_rules import HeuristicRulesEngine
from src.honey_prompt_detector.core.honey_prompt import HoneyPrompt


class SimpleContextEvaluator:
    def adjust_confidence(self, confidence, context, expected_context):
        return confidence


# ===== Benign Corpus =====
# Each category has ~50-200 samples of realistic benign text

ENGLISH_PROSE = [
    "The weather today is partly cloudy with a high of 72 degrees Fahrenheit.",
    "Machine learning models require large datasets for training and validation.",
    "The stock market experienced a significant downturn in the third quarter.",
    "Please send me the quarterly report by end of business Friday.",
    "The new restaurant on Main Street has excellent Italian cuisine.",
    "Our team completed the project two weeks ahead of schedule.",
    "The conference will be held in San Francisco from March 15 to 18.",
    "Regular exercise and a balanced diet are essential for good health.",
    "The library has an extensive collection of historical documents.",
    "Climate change is affecting weather patterns across the globe.",
    "The software update includes several bug fixes and performance improvements.",
    "Students should submit their assignments through the online portal.",
    "The museum exhibition features works from the Renaissance period.",
    "Our customer satisfaction score improved by 12% this quarter.",
    "The documentary provides an in-depth look at ocean conservation.",
    "Traffic conditions on the highway are expected to improve after 6 PM.",
    "The annual charity gala raised over $500,000 for children's education.",
    "Research indicates that sleep quality affects cognitive performance.",
    "The new policy takes effect starting January 1, 2026.",
    "Volunteers are needed for the community cleanup event this Saturday.",
    "The report summarizes findings from a three-year longitudinal study.",
    "Please review the attached document and provide your feedback.",
    "The hiking trail offers stunning views of the mountain range.",
    "Our server infrastructure was upgraded to handle increased traffic.",
    "The orchestra performed Beethoven's Symphony No. 9 to a sold-out audience.",
    "Renewable energy sources now account for 30% of electricity generation.",
    "The new employee orientation covers company policies and benefits.",
    "Archaeological excavations revealed artifacts dating back 3,000 years.",
    "The sprint retrospective identified three areas for improvement.",
    "Fresh fruits and vegetables are available at the farmers market every Sunday.",
] * 2  # 60 samples

SPANISH_TEXT = [
    "El clima de hoy es parcialmente nublado con una temperatura máxima de 22 grados.",
    "Los modelos de aprendizaje automático requieren grandes conjuntos de datos.",
    "La reunión del equipo se llevará a cabo el próximo martes a las 10 de la mañana.",
    "Por favor envíe el informe trimestral antes del viernes.",
    "El restaurante nuevo en la calle principal tiene excelente comida italiana.",
    "La conferencia sobre inteligencia artificial comienza la próxima semana.",
    "El museo presenta una exposición de arte contemporáneo latinoamericano.",
    "La educación es fundamental para el desarrollo de cualquier sociedad.",
    "Los investigadores publicaron sus hallazgos en una revista científica.",
    "El proyecto de construcción se completará en los próximos seis meses.",
    "Las energías renovables son esenciales para combatir el cambio climático.",
    "El hospital inauguró una nueva ala de emergencias la semana pasada.",
    "La biblioteca pública ofrece programas gratuitos de lectura para niños.",
    "El mercado de valores cerró con ganancias moderadas el día de hoy.",
    "Los voluntarios ayudaron a limpiar el parque después de la tormenta.",
    "La universidad anunció nuevas becas para estudiantes internacionales.",
    "El transporte público funcionará con horario reducido durante las fiestas.",
    "Los agricultores locales venden productos orgánicos en el mercado semanal.",
    "La selección nacional clasificó para la siguiente ronda del torneo.",
    "El gobierno aprobó un nuevo plan de infraestructura vial.",
] * 2  # 40 samples

CHINESE_TEXT = [
    "今天的天气预报显示明天将会有小雨。",
    "机器学习技术在医疗诊断领域取得了重大突破。",
    "这家餐厅的菜品味道非常好，服务也很周到。",
    "请在周五之前提交季度报告。",
    "新的高铁线路将连接北京和上海，行程缩短至三小时。",
    "今年的春节联欢晚会节目非常精彩。",
    "科学家们在深海中发现了一种新的生物物种。",
    "大学图书馆藏书超过五百万册，对所有学生开放。",
    "这部纪录片详细记录了大熊猫的生活习性。",
    "环保组织呼吁减少一次性塑料制品的使用。",
    "人工智能正在改变我们的工作和生活方式。",
    "今天的股市交易量创下了本月新高。",
    "博物馆的新展览吸引了大量游客参观。",
    "学校将在下学期开设新的编程课程。",
    "研究表明充足的睡眠有助于提高学习效率。",
    "城市公园在周末举办了一场文化艺术节。",
    "快递服务已经成为现代生活中不可缺少的一部分。",
    "这项新技术可以显著降低工厂的能源消耗。",
    "政府发布了新的环境保护政策。",
    "社区志愿者组织了一次义务植树活动。",
] * 2  # 40 samples

ARABIC_TEXT = [
    "الطقس اليوم مشمس مع درجة حرارة عالية تبلغ 35 درجة مئوية.",
    "يتطلب التعلم الآلي مجموعات بيانات كبيرة للتدريب والتحقق.",
    "يرجى إرسال التقرير الفصلي قبل نهاية يوم الجمعة.",
    "المطعم الجديد في الشارع الرئيسي يقدم أطباقاً عربية تقليدية ممتازة.",
    "المؤتمر السنوي للتكنولوجيا سيعقد في دبي الشهر القادم.",
    "الجامعة أعلنت عن منح دراسية جديدة للطلاب المتفوقين.",
    "المكتبة العامة توفر خدمات إلكترونية متقدمة للباحثين.",
    "فريق البحث نشر نتائج دراسته في مجلة علمية محكمة.",
    "المستشفى الجديد يوفر أحدث التقنيات الطبية.",
    "البرنامج التعليمي يهدف إلى تطوير مهارات الشباب.",
    "الحكومة أطلقت مشروعاً جديداً للطاقة المتجددة.",
    "المعرض الفني يضم أعمالاً من فنانين عرب معاصرين.",
    "الاقتصاد الوطني شهد نمواً ملحوظاً في الربع الأخير.",
    "المنظمة الخيرية جمعت تبرعات لدعم التعليم في المناطق النائية.",
    "البحث العلمي يلعب دوراً محورياً في التنمية المستدامة.",
    "النقل العام سيعمل بجدول مخفض خلال العطلات الرسمية.",
    "المزارعون يستخدمون تقنيات حديثة لزيادة إنتاج المحاصيل.",
    "الفريق الوطني تأهل للمرحلة التالية من البطولة.",
    "برنامج التدريب المهني يوفر فرص عمل للخريجين الجدد.",
    "المؤسسة أعلنت عن خطة خمسية جديدة للتطوير والتحديث.",
] * 2  # 40 samples

CODE_BLOCKS = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.describe())",
    "const express = require('express');\nconst app = express();\napp.listen(3000);",
    "SELECT u.name, u.email FROM users u JOIN orders o ON u.id = o.user_id WHERE o.total > 100;",
    "func main() {\n    fmt.Println(\"Hello, World!\")\n    http.ListenAndServe(\":8080\", nil)\n}",
    "class MyComponent extends React.Component {\n  render() {\n    return <div>Hello</div>;\n  }\n}",
    "docker build -t myapp:latest .\ndocker run -p 8080:8080 myapp:latest",
    "git checkout -b feature/new-auth\ngit add .\ngit commit -m 'Add OAuth2 support'",
    "pip install numpy pandas scikit-learn\npython -m pytest tests/ -v",
    "#!/bin/bash\nfor file in *.txt; do\n    wc -l \"$file\"\ndone",
    "async function fetchData(url) {\n  const response = await fetch(url);\n  return response.json();\n}",
    "import torch\nmodel = torch.nn.Linear(10, 5)\noptimizer = torch.optim.Adam(model.parameters())",
    "CREATE TABLE products (\n    id SERIAL PRIMARY KEY,\n    name VARCHAR(255) NOT NULL,\n    price DECIMAL(10,2)\n);",
    "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: nginx-deployment\nspec:\n  replicas: 3",
    "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}",
    "try:\n    result = process_data(input_file)\nexcept FileNotFoundError:\n    logging.error('File not found')",
    "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt",
    "const handleSubmit = (e) => {\n  e.preventDefault();\n  setLoading(true);\n  api.post('/submit', formData);\n};",
    "fn calculate_average(numbers: &[f64]) -> f64 {\n    numbers.iter().sum::<f64>() / numbers.len() as f64\n}",
    "module.exports = {\n  entry: './src/index.js',\n  output: { path: __dirname + '/dist', filename: 'bundle.js' }\n};",
] * 3  # 60 samples

JSON_STRUCTURED = [
    '{"name": "John Doe", "age": 30, "email": "john@example.com", "role": "engineer"}',
    '{"status": "success", "data": {"id": 12345, "created_at": "2025-01-15T10:30:00Z"}}',
    '{"error": {"code": 404, "message": "Resource not found", "details": "No user with ID 999"}}',
    '{"products": [{"id": 1, "name": "Widget", "price": 9.99}, {"id": 2, "name": "Gadget", "price": 19.99}]}',
    '{"config": {"database": {"host": "localhost", "port": 5432}, "cache": {"ttl": 3600}}}',
    '{"metrics": {"cpu_usage": 45.2, "memory_mb": 1024, "disk_percent": 68.5, "uptime_hours": 720}}',
    '{"user": {"first_name": "Alice", "last_name": "Smith", "preferences": {"theme": "dark", "lang": "en"}}}',
    '{"event": {"type": "click", "target": "button#submit", "timestamp": 1706000000, "page": "/checkout"}}',
    '{"order": {"id": "ORD-2025-001", "items": 3, "total": 59.97, "status": "shipped"}}',
    '{"weather": {"temp_c": 22, "humidity": 65, "wind_kph": 15, "condition": "Partly Cloudy"}}',
    '{"log": {"level": "INFO", "timestamp": "2025-01-20T14:00:00Z", "message": "Server started"}}',
    '{"response": {"status_code": 200, "headers": {"content-type": "application/json"}, "body": "{}"}}',
    '{"pipeline": {"stage": "build", "status": "passed", "duration_s": 120, "tests": {"total": 150, "passed": 148}}}',
    '{"notification": {"type": "email", "to": "user@company.com", "subject": "Weekly Report"}}',
    '{"inventory": {"sku": "ABC-123", "quantity": 500, "location": "Warehouse A", "reorder_point": 100}}',
] * 4  # 60 samples

LOG_LINES = [
    "2025-01-20 14:30:15 INFO  [main] Application started successfully on port 8080",
    "2025-01-20 14:30:16 DEBUG [db] Connection pool initialized: max=10, idle=2",
    "2025-01-20 14:30:17 WARN  [cache] Cache miss rate exceeding threshold: 45%",
    "2025-01-20 14:30:18 ERROR [auth] Failed login attempt for user admin from 192.168.1.100",
    "2025-01-20 14:30:19 INFO  [api] GET /api/v1/users 200 OK 45ms",
    "2025-01-20 14:30:20 INFO  [api] POST /api/v1/orders 201 Created 120ms",
    "2025-01-20 14:30:21 DEBUG [worker] Processing job #12345: email_notification",
    "2025-01-20 14:30:22 WARN  [memory] Heap usage at 78%, consider increasing max heap size",
    "2025-01-20 14:30:23 INFO  [scheduler] Cron job 'daily-backup' completed in 45s",
    "2025-01-20 14:30:24 ERROR [network] Connection timeout to upstream service api.example.com",
    "nginx: 192.168.1.50 - - [20/Jan/2025:14:30:25 +0000] \"GET /index.html HTTP/1.1\" 200 1234",
    "kern: [12345.678] USB device connected: vendor=0x1234 product=0x5678",
    "sshd[1234]: Accepted publickey for deploy from 10.0.0.5 port 54321 ssh2",
    "systemd[1]: Started NGINX Web Server.",
    "postgres: 2025-01-20 14:30:30 UTC LOG: checkpoint starting: time",
    "[GC (Allocation Failure) 256M->128M(512M), 0.0234 secs]",
    "celery: Task tasks.send_email[abc123] succeeded in 0.5s: None",
    "gunicorn: [2025-01-20 14:30:35] [INFO] Worker spawning (pid: 12345)",
    "redis: 20 Jan 2025 14:30:36 * Background saving started by pid 6789",
    "webpack: Compiled successfully in 2345ms",
] * 3  # 60 samples

URL_WEB_CONTENT = [
    "Visit our website at https://www.example.com/products for more information.",
    "The API documentation is available at https://docs.api.example.com/v2/reference",
    "Download the latest release from https://github.com/org/repo/releases/latest",
    "For support, contact us at support@company.com or visit https://help.company.com",
    "The blog post is available at https://blog.example.com/2025/01/machine-learning-trends",
    "Check the build status at https://ci.example.com/pipelines/12345",
    "The image is hosted at https://cdn.example.com/images/header-2025.png",
    "OAuth callback URL: https://app.example.com/auth/callback?state=abc123",
    "API endpoint: https://api.example.com/v1/users?page=2&limit=50&sort=created_at",
    "Webhook URL: https://hooks.example.com/incoming/T1234/B5678/abcdef123456",
    "The RSS feed is at https://news.example.com/feed.xml",
    "npm package: https://www.npmjs.com/package/@company/component-library",
    "Docker image: docker pull registry.example.com/myapp:v2.1.0",
    "S3 bucket URL: s3://my-bucket/data/exports/2025-01-20/report.csv",
    "Confluence page: https://wiki.company.com/display/TEAM/Sprint+Planning+2025",
] * 4  # 60 samples


def build_corpus():
    """Build the stratified benign corpus."""
    return {
        "english_prose": ENGLISH_PROSE,
        "spanish": SPANISH_TEXT,
        "chinese": CHINESE_TEXT,
        "arabic": ARABIC_TEXT,
        "code_blocks": CODE_BLOCKS,
        "json_structured": JSON_STRUCTURED,
        "log_lines": LOG_LINES,
        "urls_web": URL_WEB_CONTENT,
    }


def main():
    n_tokens = 10
    project_root = Path(__file__).resolve().parent.parent

    print("Benign FP Stratified Corpus Study")
    print("=" * 60)

    corpus = build_corpus()
    total_samples = sum(len(v) for v in corpus.values())
    print(f"Corpus: {len(corpus)} categories, {total_samples} total samples\n")

    # Initialize components
    heuristic_engine = HeuristicRulesEngine()
    context_eval = SimpleContextEvaluator()
    detector = Detector(
        context_evaluator=context_eval,
        heuristic_engine=heuristic_engine,
        attack_memory=None,
        initial_threshold=0.70,
    )

    # Generate random tokens
    tokens = [secrets.token_urlsafe(16) for _ in range(n_tokens)]

    # Results storage
    results = {}

    for cat_name, texts in corpus.items():
        stage3_fps = 0
        stage3_total = 0
        stage1_fps = 0
        stage1_total = 0
        rule_triggers = {}

        for text in texts:
            # Test with each token
            for token in tokens:
                honey_prompt = HoneyPrompt(
                    base_token=token,
                    category="security_canary",
                    sensitivity=1.0,
                    context="system security canary token",
                    variations=[token.lower(), token.upper(), token.replace("-", "_")],
                )

                # Stage 3 token matching
                result = detector.analyze_text(
                    text=text,
                    honey_prompt=honey_prompt,
                    context_window_size=200,
                    skip_heuristics=True,
                    skip_memory=True,
                )
                stage3_total += 1
                if result.get("matched", False):
                    stage3_fps += 1

            # Stage 1 heuristics (token-independent)
            stage1_total += 1
            quick_result = heuristic_engine.scan_quick(text)
            if not quick_result:
                matches = heuristic_engine.scan(text)
                quick_result = matches[0] if matches else None

            if quick_result:
                stage1_fps += 1
                rule_name = quick_result.rule_name
                rule_triggers[rule_name] = rule_triggers.get(rule_name, 0) + 1

        stage3_fpr = stage3_fps / stage3_total if stage3_total > 0 else 0
        stage1_fpr = stage1_fps / stage1_total if stage1_total > 0 else 0

        results[cat_name] = {
            "n_texts": len(texts),
            "stage3_fpr": stage3_fpr,
            "stage3_fps": stage3_fps,
            "stage3_total": stage3_total,
            "stage1_fpr": stage1_fpr,
            "stage1_fps": stage1_fps,
            "stage1_total": stage1_total,
            "rule_triggers": rule_triggers,
        }

        status_s3 = "OK" if stage3_fpr == 0 else "ALERT"
        status_s1 = "OK" if stage1_fpr < 0.05 else "HIGH" if stage1_fpr < 0.15 else "ALERT"
        print(f"  {cat_name:<20s}  S3 FPR={stage3_fpr:.4f} [{status_s3}]  "
              f"S1 FPR={stage1_fpr:.4f} [{status_s1}]  "
              f"(n={len(texts)})")
        if rule_triggers:
            top_rules = sorted(rule_triggers.items(), key=lambda x: x[1], reverse=True)[:3]
            for rule, count in top_rules:
                print(f"    -> {rule}: {count} triggers")

    # Overall
    total_s3_fps = sum(r["stage3_fps"] for r in results.values())
    total_s3 = sum(r["stage3_total"] for r in results.values())
    total_s1_fps = sum(r["stage1_fps"] for r in results.values())
    total_s1 = sum(r["stage1_total"] for r in results.values())

    print(f"\nOverall Stage 3 FPR: {total_s3_fps}/{total_s3} = {total_s3_fps/total_s3:.6f}")
    print(f"Overall Stage 1 FPR: {total_s1_fps}/{total_s1} = {total_s1_fps/total_s1:.4f}")

    # Write summary
    out_dir = project_root / "results"
    out_dir.mkdir(exist_ok=True)
    summary_path = out_dir / "experiment_benign_fp_stratified_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Benign FP Stratified Corpus Study\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Tokens per category: {n_tokens}\n")
        f.write(f"Total samples: {total_samples}\n\n")
        f.write(f"Overall Stage 3 FPR: {total_s3_fps/total_s3:.6f} ({total_s3_fps}/{total_s3})\n")
        f.write(f"Overall Stage 1 FPR: {total_s1_fps/total_s1:.4f} ({total_s1_fps}/{total_s1})\n\n")
        f.write("Per-category:\n")
        for cat_name, r in results.items():
            f.write(f"  {cat_name:<20s}  S3_FPR={r['stage3_fpr']:.6f}  "
                    f"S1_FPR={r['stage1_fpr']:.4f}  n={r['n_texts']}\n")
            if r["rule_triggers"]:
                for rule, count in sorted(r["rule_triggers"].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"    {rule}: {count}\n")

    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
