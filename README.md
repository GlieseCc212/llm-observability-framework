# 🔍 LLM Observability Framework

A comprehensive framework for monitoring, evaluating, and alerting on Large Language Model performance in production environments.

## 🌟 Features

### 📊 **Monitoring & Metrics**
- Real-time performance tracking (response times, token usage, costs)
- Sliding window analytics with anomaly detection
- Success rate and error monitoring
- Quality score tracking and trends

### 📈 **Interactive Dashboard**
- Web-based Streamlit dashboard with real-time updates
- Performance trend visualization with Plotly charts
- Model comparison and analysis tools
- Customizable time ranges and filters

### 🚨 **Intelligent Alerting**
- Configurable alert rules with multiple severity levels
- Multiple notification channels (Email, Slack, Discord, Webhooks)
- Alert throttling and cooldown mechanisms
- Historical alert analysis and statistics

### 🎯 **Advanced Evaluation**
- Multi-dimensional response quality assessment
- Bias and toxicity detection
- Hallucination detection algorithms
- Task-specific evaluation metrics (Q&A, summarization, creative writing)
- Custom evaluation framework support

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-observability-framework

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data logs
```

### Basic Usage

```python
from monitoring import MetricsCollector, PerformanceTracker
from alerts import AlertManager
from evaluation import LLMEvaluator

# Initialize components
collector = MetricsCollector()
tracker = PerformanceTracker()
alert_manager = AlertManager()
evaluator = LLMEvaluator()

# Track a request
import uuid
from datetime import datetime

request_id = str(uuid.uuid4())
model_name = "gpt-3.5-turbo"

# Example LLM request tracking
with collector.MetricsCollectorContext(collector, request_id, model_name) as metrics:
    # Simulate your LLM call
    prompt = "What is machine learning?"
    response = "Machine learning is a subset of AI..."  # Your LLM response
    
    # Update metrics
    metrics.prompt_tokens = 50
    metrics.completion_tokens = 120
    metrics.total_tokens = 170
    metrics.cost_usd = 0.0034
    
    # Track performance
    tracker.track_request(
        model_name=model_name,
        response_time_ms=metrics.response_time_ms,
        total_tokens=metrics.total_tokens,
        success=True,
        cost_usd=metrics.cost_usd
    )
    
    # Evaluate response quality
    eval_result = evaluator.evaluate_response(
        model_name=model_name,
        prompt=prompt,
        response=response,
        task_type="qa"
    )
    
    print(f"Quality Score: {eval_result.quality_score:.2f}")
    print(f"Relevance Score: {eval_result.relevance_score:.2f}")
```

### Launch Dashboard

```bash
# Run the Streamlit dashboard
streamlit run dashboard/streamlit_dashboard.py

# Or use the convenience script
python -m dashboard.streamlit_dashboard
```

Visit `http://localhost:8501` to view the dashboard.

## 📋 Core Components

### 1. **MetricsCollector**
Collects and stores comprehensive LLM performance metrics:

```python
from monitoring import MetricsCollector, LLMMetrics

collector = MetricsCollector()

# Create metrics manually
metrics = LLMMetrics(
    timestamp=datetime.now().isoformat(),
    request_id="req_123",
    model_name="gpt-4",
    prompt_tokens=100,
    completion_tokens=200,
    total_tokens=300,
    response_time_ms=1500.0,
    cost_usd=0.006,
    success=True,
    quality_score=0.85
)

collector.collect_metrics(metrics)

# Get summary statistics
stats = collector.get_summary_stats(
    start_time="2024-01-01T00:00:00",
    end_time="2024-01-31T23:59:59"
)
print(f"Average response time: {stats['avg_response_time']:.2f}ms")
print(f"Success rate: {stats['success_rate']:.2%}")
```

### 2. **PerformanceTracker**
Real-time performance monitoring with sliding window analytics:

```python
from monitoring import PerformanceTracker

def alert_callback(alert):
    print(f"ALERT: {alert.message}")

tracker = PerformanceTracker(
    window_size_minutes=15,
    alert_callback=alert_callback
)

# Track requests in real-time
tracker.track_request(
    model_name="claude-3",
    response_time_ms=2000,
    total_tokens=150,
    success=True,
    cost_usd=0.003,
    quality_score=0.92
)

# Get current performance stats
stats = tracker.get_current_stats("claude-3")
print(f"P95 Response Time: {stats['claude-3']['response_time']['p95']:.0f}ms")
```

### 3. **AlertManager**
Comprehensive alerting system with multiple notification channels:

```python
from alerts import AlertManager, AlertRule
from alerts.notification_handlers import SlackNotifier, SlackConfig

alert_manager = AlertManager()

# Add custom alert rule
custom_rule = AlertRule(
    rule_id="high_cost_alert",
    name="High Cost Per Request",
    metric_name="cost_per_token",
    condition="gt",
    threshold=0.01,
    severity="warning",
    cooldown_minutes=10,
    description="Alert when cost per token exceeds $0.01"
)
alert_manager.add_rule(custom_rule)

# Configure Slack notifications
slack_config = SlackConfig(
    webhook_url="https://hooks.slack.com/your/webhook/url",
    channel="#llm-alerts"
)
slack_notifier = SlackNotifier(slack_config)
alert_manager.add_notification_handler(slack_notifier)

# Get active alerts
active_alerts = alert_manager.get_active_alerts(severity="critical")
for alert in active_alerts:
    print(f"Active Alert: {alert['message']}")
```

### 4. **LLMEvaluator**
Advanced response quality evaluation:

```python
from evaluation import LLMEvaluator

evaluator = LLMEvaluator()

# Evaluate a response
result = evaluator.evaluate_response(
    model_name="gpt-4",
    prompt="Explain quantum computing to a 10-year-old",
    response="Quantum computing is like having a super fast computer that can solve very hard problems...",
    task_type="general",
    expected_response="A simple explanation suitable for children..."  # Optional
)

print(f"Quality Score: {result.quality_score:.2f}")
print(f"Relevance Score: {result.relevance_score:.2f}")
print(f"Coherence Score: {result.coherence_score:.2f}")
print(f"Toxicity Score: {result.toxicity_score:.2f}")
print(f"Bias Score: {result.bias_score:.2f}")

# Add custom evaluator
def my_custom_evaluator(response, prompt, custom_metrics):
    # Your custom evaluation logic
    return {"my_metric": 0.75}

evaluator.add_custom_evaluator("custom_task", my_custom_evaluator)
```

## 🎛️ Configuration

The framework uses YAML configuration files located in the `config/` directory:

```yaml
# config/config.yaml
monitoring:
  performance_tracker:
    window_size_minutes: 15
    max_data_points: 1000

alerts:
  thresholds:
    response_time_ms:
      warning: 5000
      critical: 15000
    success_rate:
      warning: 0.95
      critical: 0.85

dashboard:
  streamlit:
    port: 8501
    auto_refresh_seconds: 30
```

## 📊 Dashboard Features

The Streamlit dashboard provides:

- **Overview Metrics**: Total requests, success rates, costs, and quality scores
- **Real-time Performance**: Live tracking of current system performance  
- **Historical Trends**: Time series charts for all key metrics
- **Model Comparison**: Side-by-side performance analysis
- **Error Analysis**: Detailed breakdown of failures and issues
- **Data Export**: Download metrics in JSON/CSV formats

## 🔔 Notification Channels

### Email Notifications
```python
from alerts.notification_handlers import EmailNotifier, EmailConfig

email_config = EmailConfig(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    username="your-email@gmail.com",
    password="your-app-password",
    from_address="alerts@yourcompany.com"
)

email_notifier = EmailNotifier(
    config=email_config,
    recipients=["admin@company.com", "dev-team@company.com"]
)
```

### Slack Notifications
```python
from alerts.notification_handlers import SlackNotifier, SlackConfig

slack_config = SlackConfig(
    webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    channel="#llm-monitoring",
    username="LLM Observer"
)

slack_notifier = SlackNotifier(slack_config)
```

### Discord Notifications
```python
from alerts.notification_handlers import DiscordNotifier, DiscordConfig

discord_config = DiscordConfig(
    webhook_url="https://discord.com/api/webhooks/YOUR/WEBHOOK"
)

discord_notifier = DiscordNotifier(discord_config)
```

### Generic Webhooks
```python
from alerts.notification_handlers import WebhookNotifier, WebhookConfig

webhook_config = WebhookConfig(
    url="https://your-api.com/alerts",
    method="POST",
    headers={"Authorization": "Bearer your-token"},
    retry_count=3
)

webhook_notifier = WebhookNotifier(webhook_config)
```

## 🧪 Evaluation Metrics

### Quality Metrics
- **Quality Score**: Overall response quality assessment
- **Relevance Score**: How well the response addresses the prompt
- **Coherence Score**: Logical flow and consistency
- **Accuracy Score**: Similarity to expected response (when available)

### Safety Metrics
- **Toxicity Score**: Detection of harmful or offensive content
- **Bias Score**: Identification of biased language patterns
- **Hallucination Score**: Detection of fabricated information

### Task-Specific Metrics
- **Summarization**: Conciseness and coverage
- **Q&A**: Directness and completeness
- **Creative Writing**: Creativity and originality
- **Technical**: Accuracy and clarity

## 📁 Project Structure

```
llm-observability-framework/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── monitoring/
│   ├── __init__.py
│   ├── metrics_collector.py
│   └── performance_tracker.py
├── dashboard/
│   ├── __init__.py
│   └── streamlit_dashboard.py
├── alerts/
│   ├── __init__.py
│   ├── alert_manager.py
│   └── notification_handlers.py
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py
├── data/
│   ├── llm_metrics.db
│   ├── alerts.db
│   └── evaluations.db
├── logs/
│   └── llm_observability.log
├── tests/
│   ├── test_monitoring.py
│   ├── test_alerts.py
│   └── test_evaluation.py
└── docs/
    ├── api_reference.md
    ├── configuration.md
    └── examples/
```

## 🔧 Advanced Usage

### Custom Metrics Integration

```python
# Integrate with OpenAI
import openai
from monitoring import MetricsCollector
import time
import uuid

collector = MetricsCollector()

def track_openai_request(prompt, model="gpt-3.5-turbo"):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract metrics
        usage = response['usage']
        response_text = response['choices'][0]['message']['content']
        response_time = (time.time() - start_time) * 1000
        
        # Calculate cost (example rates)
        cost_per_token = 0.0015 / 1000  # $1.50 per 1K tokens
        cost = usage['total_tokens'] * cost_per_token
        
        # Create metrics
        metrics = LLMMetrics(
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            model_name=model,
            prompt_tokens=usage['prompt_tokens'],
            completion_tokens=usage['completion_tokens'],
            total_tokens=usage['total_tokens'],
            response_time_ms=response_time,
            cost_usd=cost,
            success=True
        )
        
        collector.collect_metrics(metrics)
        return response_text
        
    except Exception as e:
        # Track failed request
        metrics = LLMMetrics(
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            model_name=model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            response_time_ms=(time.time() - start_time) * 1000,
            cost_usd=0,
            success=False
        )
        
        collector.collect_metrics(metrics)
        raise
```

### Batch Evaluation

```python
from evaluation import LLMEvaluator
import pandas as pd

evaluator = LLMEvaluator()

# Load test dataset
test_data = pd.read_csv("test_prompts.csv")

results = []
for _, row in test_data.iterrows():
    result = evaluator.evaluate_response(
        model_name=row['model'],
        prompt=row['prompt'],
        response=row['response'],
        task_type=row['task_type'],
        expected_response=row.get('expected_response')
    )
    results.append(result)

# Analyze results
summary = evaluator.get_evaluation_summary(
    model_name="gpt-4",
    task_type="qa"
)
print(f"Average Quality: {summary['average_quality']:.2f}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: See the `docs/` directory for detailed guides
- **Issues**: Report bugs and feature requests on GitHub Issues
- **Examples**: Check `docs/examples/` for more usage examples

## 🙏 Acknowledgments

- Built with Streamlit for the dashboard interface
- Uses Plotly for interactive visualizations
- SQLite for efficient data storage
- Supports multiple LLM providers and frameworks