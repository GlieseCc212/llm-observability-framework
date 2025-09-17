#!/usr/bin/env python3
"""
Basic Usage Example for LLM Observability Framework

This example demonstrates how to:
1. Set up monitoring components
2. Track LLM requests
3. Configure alerts
4. Evaluate responses
5. View dashboard data
"""

import time
import uuid
import random
from datetime import datetime
from typing import List, Dict

# Import framework components
from monitoring import MetricsCollector, PerformanceTracker, LLMMetrics
from alerts import AlertManager, AlertRule
from alerts.notification_handlers import LogNotifier
from evaluation import LLMEvaluator


def simulate_llm_request(model_name: str, prompt: str) -> Dict:
    """Simulate an LLM request and return mock response data"""
    
    # Simulate different response times and success rates
    if "gpt-4" in model_name.lower():
        response_time = random.uniform(1000, 3000)  # 1-3 seconds
        success_rate = 0.98
        cost_per_token = 0.03 / 1000
    elif "gpt-3.5" in model_name.lower():
        response_time = random.uniform(500, 1500)   # 0.5-1.5 seconds  
        success_rate = 0.95
        cost_per_token = 0.002 / 1000
    elif "claude" in model_name.lower():
        response_time = random.uniform(800, 2000)   # 0.8-2 seconds
        success_rate = 0.96
        cost_per_token = 0.008 / 1000
    else:
        response_time = random.uniform(300, 1000)   # 0.3-1 seconds
        success_rate = 0.90
        cost_per_token = 0.001 / 1000
    
    # Simulate request success/failure
    success = random.random() < success_rate
    
    if success:
        # Generate mock response
        responses = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "Quantum computing uses quantum mechanical phenomena to perform calculations that would be impossible or impractical for classical computers.",
            "Natural language processing is a branch of AI that helps computers understand, interpret and manipulate human language.",
            "Deep learning is a neural network with three or more layers that attempts to simulate the behavior of the human brain.",
            "Computer vision is a field of AI that trains computers to interpret and understand the visual world from digital images or videos."
        ]
        response_text = random.choice(responses)
        
        # Calculate token usage
        prompt_tokens = len(prompt.split()) * 1.3  # Rough approximation
        completion_tokens = len(response_text.split()) * 1.3
        total_tokens = int(prompt_tokens + completion_tokens)
        cost = total_tokens * cost_per_token
        
        return {
            'success': True,
            'response': response_text,
            'response_time_ms': response_time,
            'prompt_tokens': int(prompt_tokens),
            'completion_tokens': int(completion_tokens),
            'total_tokens': total_tokens,
            'cost_usd': cost
        }
    else:
        # Simulate failure
        error_messages = [
            "Rate limit exceeded",
            "API key invalid",
            "Model temporarily unavailable",
            "Request timeout",
            "Content policy violation"
        ]
        
        return {
            'success': False,
            'error': random.choice(error_messages),
            'response_time_ms': response_time,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'cost_usd': 0
        }


def setup_observability_framework():
    """Set up all observability components"""
    
    print("🔧 Setting up LLM Observability Framework...")
    
    # Initialize components
    metrics_collector = MetricsCollector()
    performance_tracker = PerformanceTracker(window_size_minutes=5)
    alert_manager = AlertManager()
    evaluator = LLMEvaluator()
    
    # Configure alerting
    log_notifier = LogNotifier(log_level="WARNING")
    alert_manager.add_notification_handler(log_notifier)
    
    # Add custom alert rule for demo
    custom_rule = AlertRule(
        rule_id="demo_high_latency",
        name="Demo High Latency Alert",
        metric_name="response_time_ms",
        condition="gt",
        threshold=2500,  # 2.5 seconds
        severity="warning",
        cooldown_minutes=1,
        description="Alert when response time exceeds 2.5 seconds"
    )
    alert_manager.add_rule(custom_rule)
    
    print("✅ Framework initialized successfully!")
    return metrics_collector, performance_tracker, alert_manager, evaluator


def run_simulation(collector, tracker, alert_manager, evaluator, num_requests=50):
    """Run a simulation of LLM requests"""
    
    print(f"\n🚀 Running simulation with {num_requests} requests...")
    
    models = ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "llama-2-7b"]
    prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms",
        "How does natural language processing work?", 
        "What are the benefits of deep learning?",
        "Describe computer vision applications",
        "What is the difference between AI and ML?",
        "How do neural networks function?",
        "Explain reinforcement learning",
        "What is transfer learning?",
        "How do transformers work in NLP?"
    ]
    
    task_types = ["qa", "general", "technical"]
    
    for i in range(num_requests):
        # Select random parameters
        model_name = random.choice(models)
        prompt = random.choice(prompts)
        task_type = random.choice(task_types)
        request_id = str(uuid.uuid4())
        
        print(f"  📤 Request {i+1}/{num_requests}: {model_name}")
        
        # Simulate LLM request
        result = simulate_llm_request(model_name, prompt)
        
        # Create metrics
        metrics = LLMMetrics(
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            model_name=model_name,
            prompt_tokens=result['prompt_tokens'],
            completion_tokens=result['completion_tokens'],
            total_tokens=result['total_tokens'],
            response_time_ms=result['response_time_ms'],
            cost_usd=result['cost_usd'],
            success=result['success']
        )
        
        # Collect metrics
        collector.collect_metrics(metrics)
        
        # Track performance (triggers alerts if thresholds exceeded)
        tracker.track_request(
            model_name=model_name,
            response_time_ms=result['response_time_ms'],
            total_tokens=result['total_tokens'],
            success=result['success'],
            cost_usd=result['cost_usd']
        )
        
        # Evaluate successful responses
        if result['success']:
            eval_result = evaluator.evaluate_response(
                model_name=model_name,
                prompt=prompt,
                response=result['response'],
                task_type=task_type
            )
            
            # Update metrics with quality score
            metrics.quality_score = eval_result.quality_score
            collector.collect_metrics(metrics)  # Update with quality score
        
        # Small delay to simulate real-world timing
        time.sleep(0.1)
    
    print("✅ Simulation completed!")


def display_results(collector, tracker, alert_manager, evaluator):
    """Display comprehensive results from the simulation"""
    
    print("\n📊 SIMULATION RESULTS")
    print("=" * 50)
    
    # Metrics Summary
    print("\n📈 Metrics Summary:")
    summary = collector.get_summary_stats()
    
    if summary:
        print(f"  Total Requests: {summary.get('total_requests', 0):,}")
        print(f"  Success Rate: {summary.get('success_rate', 0):.2%}")
        print(f"  Avg Response Time: {summary.get('avg_response_time', 0):.0f}ms")
        print(f"  Total Cost: ${summary.get('total_cost', 0):.4f}")
        print(f"  Avg Quality Score: {summary.get('avg_quality_score', 0) or 0:.2f}")
        
        # Model breakdown
        print(f"\n  Model Breakdown:")
        for model_data in summary.get('model_breakdown', []):
            print(f"    {model_data['model_name']}: {model_data['requests']} requests, "
                  f"${model_data['total_cost']:.4f} cost")
    else:
        print("  No metrics data available")
    
    # Real-time Performance
    print("\n⚡ Real-time Performance:")
    current_stats = tracker.get_current_stats()
    
    for model_name, stats in current_stats.items():
        print(f"  {model_name}:")
        if 'response_time' in stats:
            print(f"    P95 Response Time: {stats['response_time']['p95']:.0f}ms")
            print(f"    Avg Response Time: {stats['response_time']['avg']:.0f}ms")
        if 'success_rate' in stats:
            print(f"    Success Rate: {stats['success_rate']:.2%}")
        if 'cost' in stats:
            print(f"    Total Cost: ${stats['cost']['total']:.4f}")
    
    # Alert Statistics
    print("\n🚨 Alert Summary:")
    alert_stats = alert_manager.get_alert_statistics()
    
    if alert_stats:
        print(f"  Total Alerts: {alert_stats.get('total_alerts', 0)}")
        print(f"  Resolved Alerts: {alert_stats.get('resolved_alerts', 0)}")
        
        # Severity breakdown
        severity_breakdown = alert_stats.get('severity_breakdown', [])
        if severity_breakdown:
            print("  Alert Severity Breakdown:")
            for severity_data in severity_breakdown:
                print(f"    {severity_data['severity'].title()}: {severity_data['count']}")
    else:
        print("  No alerts generated")
    
    # Evaluation Summary
    print("\n🎯 Evaluation Summary:")
    eval_summary = evaluator.get_evaluation_summary()
    
    if eval_summary:
        print(f"  Total Evaluations: {eval_summary.get('total_evaluations', 0)}")
        print(f"  Avg Quality: {eval_summary.get('average_quality', 0) or 0:.2f}")
        print(f"  Avg Relevance: {eval_summary.get('average_relevance', 0) or 0:.2f}")
        print(f"  Avg Coherence: {eval_summary.get('average_coherence', 0) or 0:.2f}")
        
        # Quality distribution
        quality_dist = eval_summary.get('quality_distribution', {})
        if quality_dist:
            print("  Quality Score Distribution:")
            for range_key, count in quality_dist.items():
                print(f"    {range_key}: {count} responses")
    else:
        print("  No evaluation data available")


def demonstrate_dashboard_integration():
    """Demonstrate how to launch the dashboard"""
    
    print("\n🖥️ Dashboard Integration:")
    print("  To view the interactive dashboard, run:")
    print("    streamlit run dashboard/streamlit_dashboard.py")
    print("  Or:")
    print("    python -m dashboard.streamlit_dashboard")
    print("  Then visit: http://localhost:8501")


def main():
    """Main execution function"""
    
    print("🔍 LLM Observability Framework - Basic Usage Example")
    print("=" * 60)
    
    # Set up framework
    collector, tracker, alert_manager, evaluator = setup_observability_framework()
    
    # Run simulation
    run_simulation(collector, tracker, alert_manager, evaluator, num_requests=30)
    
    # Display results
    display_results(collector, tracker, alert_manager, evaluator)
    
    # Dashboard information
    demonstrate_dashboard_integration()
    
    print(f"\n✨ Example completed! Check the 'data/' directory for generated databases.")
    print("You can now:")
    print("  1. Launch the dashboard to visualize the data")
    print("  2. Query the databases directly")
    print("  3. Export metrics using the collector's export_metrics() method")
    print("  4. Set up production alerting with real notification channels")


if __name__ == "__main__":
    main()