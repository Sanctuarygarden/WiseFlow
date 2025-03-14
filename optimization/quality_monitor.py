from typing import Dict, Any, List, Optional, Tuple
import logging
import time
from datetime import datetime, timedelta
import statistics
import json

logger = logging.getLogger(__name__)

class QualityMonitor:
    """
    System self-monitoring for quality and performance
    
    This component:
    1. Tracks quality metrics over time
    2. Detects quality degradation
    3. Identifies problematic components
    4. Provides actionable insights for improvement
    """
    
    def __init__(self, alert_threshold: float = 0.2, window_size: int = 10):
        """
        Initialize quality monitor
        
        Args:
            alert_threshold: Threshold for quality degradation alerts
            window_size: Size of window for trend analysis
        """
        self.alert_threshold = alert_threshold
        self.window_size = window_size
        self.metrics = {
            "context_retention": [],  # % of core concepts retained
            "logical_coherence": [],  # coherence scores
            "concept_evolution": [],  # # of concept evolutions detected
            "concept_recalls": [],    # # of concepts recalled
            "execution_success": [],  # % of code executions succeeded
            "response_time": []       # response generation time
        }
        self.component_health = {}  # component -> [health scores]
        self.alerts = []  # List of active alerts
        self.stats = {
            "metrics_recorded": 0,
            "alerts_generated": 0,
            "checks_performed": 0
        }
        
    def record_metric(self, metric_name: str, value: float) -> bool:
        """
        Record a quality metric
        
        Args:
            metric_name: Name of metric to record
            value: Metric value
            
        Returns:
            Success flag
        """
        if metric_name not in self.metrics:
            logger.warning(f"Unknown metric name: {metric_name}")
            return False
            
        self.metrics[metric_name].append({
            "value": value,
            "timestamp": datetime.now().isoformat()
        })
        
        self.stats["metrics_recorded"] += 1
        
        # Check for quality issues after each new metric
        if self.stats["metrics_recorded"] % 5 == 0:
            self.check_quality()
            
        return True
        
    def record_component_health(self, component_name: str, health_score: float) -> bool:
        """
        Record component health score
        
        Args:
            component_name: Name of component
            health_score: Health score (0.0-1.0)
            
        Returns:
            Success flag
        """
        if component_name not in self.component_health:
            self.component_health[component_name] = []
            
        self.component_health[component_name].append({
            "score": health_score,
            "timestamp": datetime.now().isoformat()
        })
        
        return True
        
    def check_quality(self) -> Dict[str, Any]:
        """
        Check system quality and generate alerts
        
        Returns:
            Quality check results
        """
        self.stats["checks_performed"] += 1
        current_time = datetime.now()
        
        # Analyze trends for each metric
        trends = {}
        issues = []
        
        for metric_name, values in self.metrics.items():
            if len(values) < self.window_size:
                trends[metric_name] = "insufficient_data"
                continue
                
            trend, severity = self._analyze_trend(values[-self.window_size:])
            trends[metric_name] = trend
            
            # Check for issues
            if trend == "degrading" and severity >= self.alert_threshold:
                issues.append({
                    "metric": metric_name,
                    "trend": trend,
                    "severity": severity,
                    "detected_at": current_time.isoformat()
                })
        
        # Analyze component health
        component_issues = []
        for component, values in self.component_health.items():
            if len(values) < 3:  # Need at least 3 points for trend
                continue
                
            recent_scores = [v["score"] for v in values[-3:]]
            avg_score = sum(recent_scores) / len(recent_scores)
            
            if avg_score < 0.7:  # Arbitrary threshold
                component_issues.append({
                    "component": component,
                    "health_score": avg_score,
                    "detected_at": current_time.isoformat()
                })
        
        # Generate alerts for new issues
        new_alerts = 0
        for issue in issues:
            # Check if already alerted
            if not any(a["metric"] == issue["metric"] and a["resolved_at"] is None for a in self.alerts):
                self.alerts.append({
                    "type": "metric_degradation",
                    "metric": issue["metric"],
                    "severity": issue["severity"],
                    "detected_at": issue["detected_at"],
                    "resolved_at": None
                })
                new_alerts += 1
                logger.warning(f"Quality alert: {issue['metric']} is degrading (severity: {issue['severity']:.2f})")
        
        for issue in component_issues:
            # Check if already alerted
            if not any(a["component"] == issue["component"] and a["resolved_at"] is None 
                     for a in self.alerts if "component" in a):
                self.alerts.append({
                    "type": "component_health",
                    "component": issue["component"],
                    "health_score": issue["health_score"],
                    "detected_at": issue["detected_at"],
                    "resolved_at": None
                })
                new_alerts += 1
                logger.warning(f"Component health alert: {issue['component']} score is {issue['health_score']:.2f}")
        
        self.stats["alerts_generated"] += new_alerts
        
        # Auto-resolve old alerts that have improved
        self._resolve_improved_alerts()
        
        return {
            "trends": trends,
            "metric_issues": issues,
            "component_issues": component_issues,
            "active_alerts": len([a for a in self.alerts if a["resolved_at"] is None]),
            "new_alerts": new_alerts
        }
        
    def get_trend(self, metric_name: str, window: Optional[int] = None) -> Tuple[str, float]:
        """
        Get trend for a specific metric
        
        Args:
            metric_name: Metric name
            window: Optional window size
            
        Returns:
            Tuple of (trend, severity)
        """
        if metric_name not in self.metrics:
            return "unknown", 0.0
            
        if window is None:
            window = self.window_size
            
        values = self.metrics[metric_name]
        
        if len(values) < window:
            return "insufficient_data", 0.0
            
        return self._analyze_trend(values[-window:])
        
    def _analyze_trend(self, data_points: List[Dict[str, Any]]) -> Tuple[str, float]:
        """
        Analyze trend in a series of data points
        
        Args:
            data_points: List of data points with value and timestamp
            
        Returns:
            Tuple of (trend_type, severity)
        """
        if not data_points or len(data_points) < 2:
            return "insufficient_data", 0.0
            
        # Extract values
        values = [point["value"] for point in data_points]
        
        # Split into first and second half
        mid_point = len(values) // 2
        first_half = values[:mid_point]
        second_half = values[mid_point:]
        
        # Calculate averages
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        # Calculate change percentage
        if first_avg == 0:
            change_pct = 1.0 if second_avg > 0 else 0.0
        else:
            change_pct = (second_avg - first_avg) / abs(first_avg)
        
        # Determine trend
        if abs(change_pct) < 0.05:  # Less than 5% change
            return "stable", abs(change_pct)
        elif change_pct > 0:
            return "improving", change_pct
        else:
            return "degrading", abs(change_pct)
            
    def _resolve_improved_alerts(self) -> None:
        """Automatically resolve alerts that have improved"""
        current_time = datetime.now()
        
        for alert in self.alerts:
            if alert["resolved_at"] is not None:
                continue  # Already resolved
                
            if alert["type"] == "metric_degradation":
                metric = alert["metric"]
                trend, _ = self.get_trend(metric, window=5)  # Check recent trend
                
                if trend == "improving" or trend == "stable":
                    alert["resolved_at"] = current_time.isoformat()
                    logger.info(f"Automatically resolved alert for {metric} - trend is now {trend}")
            
            elif alert["type"] == "component_health":
                component = alert["component"]
                
                if component in self.component_health and len(self.component_health[component]) >= 2:
                    recent_score = self.component_health[component][-1]["score"]
                    
                    if recent_score >= 0.8:  # Arbitrary threshold for "good health"
                        alert["resolved_at"] = current_time.isoformat()
                        logger.info(f"Automatically resolved alert for {component} - health score now {recent_score:.2f}")
    
    def get_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive quality report
        
        Returns:
            Quality report with trends and issues
        """
        # Analyze all metrics
        metric_trends = {}
        for metric_name in self.metrics:
            trend, severity = self.get_trend(metric_name)
            metric_trends[metric_name] = {
                "trend": trend,
                "severity": severity,
                "data_points": len(self.metrics[metric_name])
            }
        
        # Analyze all components
        component_health = {}
        for component, values in self.component_health.items():
            if not values:
                continue
                
            recent_values = values[-min(5, len(values)):]
            avg_score = sum(v["score"] for v in recent_values) / len(recent_values)
            
            component_health[component] = {
                "current_score": values[-1]["score"],
                "average_score": avg_score,
                "status": "healthy" if avg_score >= 0.7 else "degraded"
            }
        
        # Get active alerts
        active_alerts = [a for a in self.alerts if a["resolved_at"] is None]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metric_trends, component_health, active_alerts)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": self._calculate_overall_health(),
            "metric_trends": metric_trends,
            "component_health": component_health,
            "active_alerts": active_alerts,
            "resolved_alerts_count": len(self.alerts) - len(active_alerts),
            "recommendations": recommendations
        }
        
    def _calculate_overall_health(self) -> float:
        """Calculate overall system health score"""
        # If no metrics recorded, default to 0.5
        if not any(self.metrics.values()) and not any(self.component_health.values()):
            return 0.5
            
        health_scores = []
        
        # Calculate normalized metric health (degrading metrics reduce score)
        for metric, values in self.metrics.items():
            if not values:
                continue
                
            trend, severity = self.get_trend(metric)
            if trend == "degrading":
                health_scores.append(1.0 - min(severity, 0.5))  # Cap severity impact
            elif trend == "improving":
                health_scores.append(0.8)  # Good but not perfect
            else:  # stable or insufficient_data
                health_scores.append(0.7)  # Neutral score
        
        # Factor in component health
        for component, values in self.component_health.items():
            if not values:
                continue
                
            health_scores.append(values[-1]["score"])
        
        # Calculate average health score
        if not health_scores:
            return 0.5
            
        return sum(health_scores) / len(health_scores)
        
    def _generate_recommendations(
        self,
        metric_trends: Dict[str, Any],
        component_health: Dict[str, Any],
        active_alerts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on trends and health"""
        recommendations = []
        
        # Check for degrading metrics
        for metric, data in metric_trends.items():
            if data["trend"] == "degrading" and data["severity"] >= 0.1:
                if metric == "context_retention":
                    recommendations.append({
                        "priority": "high" if data["severity"] >= 0.2 else "medium",
                        "component": "Context Consolidation Layer",
                        "action": "Adjust context retention thresholds or increase concept importance scores for core concepts",
                        "metric": metric
                    })
                elif metric == "logical_coherence":
                    recommendations.append({
                        "priority": "high" if data["severity"] >= 0.2 else "medium",
                        "component": "Logical Coherence Validator",
                        "action": "Review and optimize the logical validation process, consider enabling Tree-of-Thought validation",
                        "metric": metric
                    })
                elif metric == "execution_success":
                    recommendations.append({
                        "priority": "high" if data["severity"] >= 0.2 else "medium",
                        "component": "PAL-Enhanced Code Generator",
                        "action": "Review execution environment and refine code generation templates",
                        "metric": metric
                    })
        
        # Check for unhealthy components
        for component, data in component_health.items():
            if data["status"] == "degraded":
                recommendations.append({
                    "priority": "high" if data["average_score"] < 0.5 else "medium",
                    "component": component,
                    "action": f"Investigate issues with {component}, current health score is {data['current_score']:.2f}",
                    "health_score": data["current_score"]
                })
        
        # Add recommendations for active alerts
        for alert in active_alerts:
            if alert["type"] == "metric_degradation" and not any(r["metric"] == alert["metric"] for r in recommendations if "metric" in r):
                recommendations.append({
                    "priority": "high" if alert["severity"] >= 0.3 else "medium",
                    "issue": f"Active alert for degrading {alert['metric']}",
                    "action": f"Investigate and address {alert['metric']} degradation",
                    "alert_id": self.alerts.index(alert)
                })
        
        return recommendations
    
    def get_stats(self) -> Dict[str, int]:
        """Get monitoring statistics"""
        return self.stats.copy()
