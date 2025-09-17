"""
LLMEvaluator for LLM Observability Framework

Comprehensive evaluation framework for LLM performance including:
- Response quality evaluation
- Accuracy measurement
- Bias detection
- Hallucination detection
- Task-specific evaluations
- Human evaluation integration
"""

import logging
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import statistics
import asyncio
from concurrent.futures import ThreadPoolExecutor


@dataclass
class EvaluationResult:
    """Evaluation result data"""
    eval_id: str
    model_name: str
    prompt: str
    response: str
    timestamp: str
    task_type: str
    
    # Quality metrics
    quality_score: float
    relevance_score: float
    coherence_score: float
    accuracy_score: Optional[float] = None
    
    # Safety metrics
    toxicity_score: Optional[float] = None
    bias_score: Optional[float] = None
    hallucination_score: Optional[float] = None
    
    # Task-specific metrics
    task_specific_scores: Optional[Dict[str, float]] = None
    
    # Reference data
    expected_response: Optional[str] = None
    reference_score: Optional[float] = None
    
    # Evaluation metadata
    evaluator_type: str = "automated"
    evaluator_version: str = "1.0"
    evaluation_time_ms: Optional[float] = None


class LLMEvaluator:
    """Comprehensive LLM performance evaluator"""
    
    def __init__(self, db_path: str = "data/evaluations.db"):
        """
        Initialize LLMEvaluator
        
        Args:
            db_path: Path to SQLite database for storing evaluations
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Custom evaluators
        self.custom_evaluators: Dict[str, Callable] = {}
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for evaluation storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS evaluations (
                    eval_id TEXT PRIMARY KEY,
                    model_name TEXT,
                    prompt TEXT,
                    response TEXT,
                    timestamp TEXT,
                    task_type TEXT,
                    quality_score REAL,
                    relevance_score REAL,
                    coherence_score REAL,
                    accuracy_score REAL,
                    toxicity_score REAL,
                    bias_score REAL,
                    hallucination_score REAL,
                    task_specific_scores TEXT,
                    expected_response TEXT,
                    reference_score REAL,
                    evaluator_type TEXT,
                    evaluator_version TEXT,
                    evaluation_time_ms REAL
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_eval_timestamp ON evaluations(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_eval_model ON evaluations(model_name)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_eval_task ON evaluations(task_type)')
    
    def evaluate_response(self,
                         model_name: str,
                         prompt: str,
                         response: str,
                         task_type: str = "general",
                         expected_response: Optional[str] = None,
                         custom_metrics: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """
        Evaluate an LLM response comprehensively
        
        Args:
            model_name: Name of the LLM model
            prompt: Input prompt
            response: Model response
            task_type: Type of task (general, qa, summarization, etc.)
            expected_response: Expected/reference response
            custom_metrics: Custom evaluation metrics
            
        Returns:
            EvaluationResult object
        """
        import uuid
        start_time = datetime.now()
        
        # Generate unique evaluation ID
        eval_id = str(uuid.uuid4())
        
        # Perform evaluations
        quality_score = self._evaluate_quality(response, prompt)
        relevance_score = self._evaluate_relevance(response, prompt)
        coherence_score = self._evaluate_coherence(response)
        
        # Optional evaluations
        accuracy_score = None
        if expected_response:
            accuracy_score = self._evaluate_accuracy(response, expected_response)
        
        toxicity_score = self._evaluate_toxicity(response)
        bias_score = self._evaluate_bias(response)
        hallucination_score = self._evaluate_hallucination(response, prompt)
        
        # Task-specific evaluations
        task_specific_scores = self._evaluate_task_specific(
            response, prompt, task_type, custom_metrics
        )
        
        # Calculate evaluation time
        eval_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Create result object
        result = EvaluationResult(
            eval_id=eval_id,
            model_name=model_name,
            prompt=prompt,
            response=response,
            timestamp=start_time.isoformat(),
            task_type=task_type,
            quality_score=quality_score,
            relevance_score=relevance_score,
            coherence_score=coherence_score,
            accuracy_score=accuracy_score,
            toxicity_score=toxicity_score,
            bias_score=bias_score,
            hallucination_score=hallucination_score,
            task_specific_scores=task_specific_scores,
            expected_response=expected_response,
            evaluator_type="automated",
            evaluator_version="1.0",
            evaluation_time_ms=eval_time
        )
        
        # Store result
        self._store_evaluation(result)
        
        return result
    
    def _evaluate_quality(self, response: str, prompt: str) -> float:
        """Evaluate overall response quality"""
        score = 0.0
        
        # Length appropriateness (not too short, not too long)
        response_len = len(response.split())
        if 10 <= response_len <= 500:
            score += 0.2
        elif 5 <= response_len <= 1000:
            score += 0.1
        
        # Grammar and structure (basic checks)
        if self._has_proper_punctuation(response):
            score += 0.2
        
        if self._has_proper_capitalization(response):
            score += 0.2
        
        # Content quality indicators
        if self._contains_specific_information(response):
            score += 0.2
        
        # Avoid repetition
        if not self._has_excessive_repetition(response):
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_relevance(self, response: str, prompt: str) -> float:
        """Evaluate how relevant the response is to the prompt"""
        # Simple keyword-based relevance scoring
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        prompt_words -= stop_words
        response_words -= stop_words
        
        if not prompt_words:
            return 0.5  # Neutral score if no meaningful words in prompt
        
        # Calculate overlap
        overlap = len(prompt_words & response_words)
        relevance = overlap / len(prompt_words)
        
        # Boost score if response directly addresses question words
        question_words = {'what', 'when', 'where', 'who', 'why', 'how'}
        if any(word in prompt.lower() for word in question_words):
            if any(word in response.lower() for word in ['because', 'since', 'due to', 'as a result']):
                relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _evaluate_coherence(self, response: str) -> float:
        """Evaluate response coherence and flow"""
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return 0.8  # Single sentence assumed coherent
        
        score = 0.0
        
        # Check for logical flow indicators
        transition_words = ['however', 'therefore', 'moreover', 'furthermore', 'consequently', 'additionally', 'meanwhile', 'nevertheless']
        if any(word in response.lower() for word in transition_words):
            score += 0.3
        
        # Check for consistent tense usage
        if self._has_consistent_tense(sentences):
            score += 0.2
        
        # Check for topic consistency (basic)
        if self._maintains_topic_consistency(sentences):
            score += 0.3
        
        # Penalize for contradictions (basic check)
        if not self._contains_contradictions(response):
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_accuracy(self, response: str, expected_response: str) -> float:
        """Evaluate accuracy against expected response"""
        # Simple similarity-based accuracy
        response_words = set(response.lower().split())
        expected_words = set(expected_response.lower().split())
        
        if not expected_words:
            return 0.5
        
        # Calculate Jaccard similarity
        intersection = len(response_words & expected_words)
        union = len(response_words | expected_words)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _evaluate_toxicity(self, response: str) -> float:
        """Evaluate response toxicity (basic implementation)"""
        # Basic keyword-based toxicity detection
        toxic_words = [
            'hate', 'kill', 'die', 'stupid', 'idiot', 'dumb', 'pathetic', 
            'loser', 'worthless', 'disgusting', 'terrible', 'awful'
        ]
        
        response_lower = response.lower()
        toxic_count = sum(1 for word in toxic_words if word in response_lower)
        
        if toxic_count == 0:
            return 0.0
        
        # Normalize by response length
        toxicity = min(toxic_count / len(response.split()) * 10, 1.0)
        return toxicity
    
    def _evaluate_bias(self, response: str) -> float:
        """Evaluate potential bias in response (basic implementation)"""
        # Check for potentially biased language patterns
        bias_indicators = [
            'all women', 'all men', 'all people of', 'always', 'never',
            'obviously', 'clearly', 'everyone knows', 'it\'s common sense'
        ]
        
        response_lower = response.lower()
        bias_count = sum(1 for indicator in bias_indicators if indicator in response_lower)
        
        if bias_count == 0:
            return 0.0
        
        return min(bias_count / 10.0, 1.0)
    
    def _evaluate_hallucination(self, response: str, prompt: str) -> float:
        """Evaluate potential hallucination in response"""
        # Basic hallucination detection
        hallucination_score = 0.0
        
        # Check for specific claims without context in prompt
        specific_numbers = re.findall(r'\b\d{4}\b|\b\d+%\b|\$\d+', response)
        prompt_numbers = re.findall(r'\b\d{4}\b|\b\d+%\b|\$\d+', prompt)
        
        # Numbers in response not in prompt might be hallucinated
        hallucinated_numbers = [num for num in specific_numbers if num not in prompt_numbers]
        if hallucinated_numbers:
            hallucination_score += min(len(hallucinated_numbers) * 0.2, 0.6)
        
        # Check for definitive statements about unverifiable claims
        definitive_words = ['definitely', 'certainly', 'absolutely', 'without doubt']
        if any(word in response.lower() for word in definitive_words):
            hallucination_score += 0.2
        
        return min(hallucination_score, 1.0)
    
    def _evaluate_task_specific(self, 
                              response: str, 
                              prompt: str, 
                              task_type: str,
                              custom_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Evaluate task-specific metrics"""
        scores = {}
        
        if task_type == "summarization":
            scores["conciseness"] = self._evaluate_conciseness(response)
            scores["coverage"] = self._evaluate_coverage(response, prompt)
            
        elif task_type == "qa":
            scores["directness"] = self._evaluate_directness(response, prompt)
            scores["completeness"] = self._evaluate_completeness(response, prompt)
            
        elif task_type == "creative":
            scores["creativity"] = self._evaluate_creativity(response)
            scores["originality"] = self._evaluate_originality(response)
            
        elif task_type == "technical":
            scores["technical_accuracy"] = self._evaluate_technical_accuracy(response)
            scores["clarity"] = self._evaluate_technical_clarity(response)
        
        # Apply custom metrics if provided
        if custom_metrics and task_type in self.custom_evaluators:
            custom_scores = self.custom_evaluators[task_type](response, prompt, custom_metrics)
            scores.update(custom_scores)
        
        return scores
    
    # Helper methods for basic text analysis
    def _has_proper_punctuation(self, text: str) -> bool:
        """Check if text has proper punctuation"""
        sentences = re.split(r'[.!?]+', text)
        return len(sentences) > 1 or any(p in text for p in '.!?')
    
    def _has_proper_capitalization(self, text: str) -> bool:
        """Check if text has proper capitalization"""
        sentences = re.split(r'[.!?]+', text.strip())
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence[0].isupper():
                return False
        return True
    
    def _contains_specific_information(self, text: str) -> bool:
        """Check if text contains specific, informative content"""
        # Look for numbers, proper nouns, or specific terms
        has_numbers = bool(re.search(r'\d+', text))
        has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+\b', text))
        has_specific_terms = len(text.split()) > 5
        
        return has_numbers or has_proper_nouns or has_specific_terms
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """Check for excessive word repetition"""
        words = text.lower().split()
        if len(words) < 10:
            return False
        
        # Check for repeated phrases
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Consider repetitive if any word appears more than 20% of the time
        max_count = max(word_counts.values())
        return max_count > len(words) * 0.2
    
    def _has_consistent_tense(self, sentences: List[str]) -> bool:
        """Basic tense consistency check"""
        # This is a simplified implementation
        past_indicators = ['was', 'were', 'had', 'did', 'ed']
        present_indicators = ['is', 'are', 'has', 'does']
        future_indicators = ['will', 'shall', 'going to']
        
        tense_scores = {'past': 0, 'present': 0, 'future': 0}
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in past_indicators):
                tense_scores['past'] += 1
            if any(indicator in sentence_lower for indicator in present_indicators):
                tense_scores['present'] += 1
            if any(indicator in sentence_lower for indicator in future_indicators):
                tense_scores['future'] += 1
        
        # Check if one tense dominates
        total_tense_markers = sum(tense_scores.values())
        if total_tense_markers == 0:
            return True  # No clear tense markers, assume consistent
        
        dominant_tense = max(tense_scores.values())
        return dominant_tense / total_tense_markers > 0.6
    
    def _maintains_topic_consistency(self, sentences: List[str]) -> bool:
        """Basic topic consistency check"""
        # Simple implementation: check for shared keywords
        if len(sentences) < 2:
            return True
        
        # Get keywords from each sentence
        sentence_keywords = []
        for sentence in sentences:
            words = [word.lower() for word in sentence.split() if len(word) > 3]
            sentence_keywords.append(set(words))
        
        # Check for overlap between consecutive sentences
        overlaps = []
        for i in range(len(sentence_keywords) - 1):
            overlap = len(sentence_keywords[i] & sentence_keywords[i + 1])
            overlaps.append(overlap)
        
        # Consider consistent if average overlap is > 0
        return statistics.mean(overlaps) > 0 if overlaps else True
    
    def _contains_contradictions(self, text: str) -> bool:
        """Basic contradiction detection"""
        # Look for contradictory patterns
        contradiction_patterns = [
            (r'\byes\b.*\bno\b', r'\bno\b.*\byes\b'),
            (r'\btrue\b.*\bfalse\b', r'\bfalse\b.*\btrue\b'),
            (r'\bgood\b.*\bbad\b', r'\bbad\b.*\bgood\b'),
            (r'\bpositive\b.*\bnegative\b', r'\bnegative\b.*\bpositive\b')
        ]
        
        text_lower = text.lower()
        for pattern1, pattern2 in contradiction_patterns:
            if re.search(pattern1, text_lower) or re.search(pattern2, text_lower):
                return True
        
        return False
    
    # Task-specific evaluation methods
    def _evaluate_conciseness(self, response: str) -> float:
        """Evaluate conciseness for summarization tasks"""
        word_count = len(response.split())
        # Ideal summary length: 50-200 words
        if 50 <= word_count <= 200:
            return 1.0
        elif word_count < 50:
            return word_count / 50.0
        else:
            return max(0.2, 1.0 - (word_count - 200) / 1000.0)
    
    def _evaluate_coverage(self, response: str, prompt: str) -> float:
        """Evaluate coverage of key points"""
        # Extract potential key points from prompt (simplified)
        prompt_sentences = re.split(r'[.!?]+', prompt)
        key_concepts = set()
        
        for sentence in prompt_sentences:
            words = [word.lower() for word in sentence.split() if len(word) > 4]
            key_concepts.update(words)
        
        if not key_concepts:
            return 0.5
        
        response_words = set(word.lower() for word in response.split())
        covered = len(key_concepts & response_words)
        
        return covered / len(key_concepts)
    
    def _evaluate_directness(self, response: str, prompt: str) -> float:
        """Evaluate directness for Q&A tasks"""
        # Check if response directly addresses the question
        question_words = ['what', 'when', 'where', 'who', 'why', 'how']
        prompt_lower = prompt.lower()
        
        for q_word in question_words:
            if q_word in prompt_lower:
                # Look for direct answers in response
                if any(phrase in response.lower() for phrase in [
                    'the answer is', 'it is', 'this is', 'because', 'due to'
                ]):
                    return 0.8
                break
        
        # Check if response starts directly (not with filler)
        response_start = response.lower().strip()
        filler_starts = ['well', 'so', 'actually', 'basically', 'essentially']
        
        if not any(response_start.startswith(filler) for filler in filler_starts):
            return 0.6
        
        return 0.3
    
    def _evaluate_completeness(self, response: str, prompt: str) -> float:
        """Evaluate completeness of answer"""
        # Basic completeness check based on response length and structure
        word_count = len(response.split())
        
        if word_count < 10:
            return 0.2
        elif word_count < 50:
            return 0.6
        else:
            return 1.0
    
    def _evaluate_creativity(self, response: str) -> float:
        """Evaluate creativity of response"""
        # Basic creativity indicators
        creativity_score = 0.0
        
        # Check for metaphors or analogies
        metaphor_words = ['like', 'as if', 'similar to', 'reminds me of', 'metaphorically']
        if any(word in response.lower() for word in metaphor_words):
            creativity_score += 0.3
        
        # Check for descriptive language
        descriptive_words = ['vivid', 'beautiful', 'stunning', 'amazing', 'incredible']
        if any(word in response.lower() for word in descriptive_words):
            creativity_score += 0.2
        
        # Check for unique word combinations
        words = response.lower().split()
        unique_combinations = len(set(zip(words, words[1:])))
        creativity_score += min(unique_combinations / 50.0, 0.5)
        
        return min(creativity_score, 1.0)
    
    def _evaluate_originality(self, response: str) -> float:
        """Evaluate originality of response"""
        # Basic originality check (avoiding common phrases)
        common_phrases = [
            'in conclusion', 'to sum up', 'as we can see', 'it is clear that',
            'there is no doubt', 'everyone knows', 'it goes without saying'
        ]
        
        response_lower = response.lower()
        common_count = sum(1 for phrase in common_phrases if phrase in response_lower)
        
        # Higher originality = fewer common phrases
        originality = max(0.0, 1.0 - common_count * 0.2)
        
        return originality
    
    def _evaluate_technical_accuracy(self, response: str) -> float:
        """Evaluate technical accuracy (basic implementation)"""
        # Look for technical terms and proper usage
        technical_indicators = [
            'algorithm', 'function', 'variable', 'parameter', 'method',
            'class', 'object', 'instance', 'framework', 'library'
        ]
        
        response_lower = response.lower()
        tech_count = sum(1 for term in technical_indicators if term in response_lower)
        
        if tech_count == 0:
            return 0.5  # Neutral if no technical content
        
        # Assume higher technical term usage indicates accuracy
        return min(tech_count / 10.0, 1.0)
    
    def _evaluate_technical_clarity(self, response: str) -> float:
        """Evaluate clarity of technical explanations"""
        clarity_score = 0.0
        
        # Check for explanation patterns
        explanation_patterns = [
            'this means', 'in other words', 'for example', 'specifically',
            'to put it simply', 'basically', 'essentially'
        ]
        
        response_lower = response.lower()
        if any(pattern in response_lower for pattern in explanation_patterns):
            clarity_score += 0.5
        
        # Check for step-by-step structure
        step_indicators = ['first', 'second', 'then', 'next', 'finally', 'step']
        step_count = sum(1 for indicator in step_indicators if indicator in response_lower)
        clarity_score += min(step_count * 0.1, 0.5)
        
        return min(clarity_score, 1.0)
    
    def add_custom_evaluator(self, task_type: str, evaluator_func: Callable) -> None:
        """
        Add custom evaluator for specific task types
        
        Args:
            task_type: Type of task to evaluate
            evaluator_func: Function that takes (response, prompt, custom_metrics) and returns Dict[str, float]
        """
        self.custom_evaluators[task_type] = evaluator_func
        self.logger.info(f"Added custom evaluator for task type: {task_type}")
    
    def _store_evaluation(self, result: EvaluationResult) -> None:
        """Store evaluation result in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result_dict = asdict(result)
                
                # Convert task_specific_scores to JSON
                if result_dict['task_specific_scores']:
                    result_dict['task_specific_scores'] = json.dumps(result_dict['task_specific_scores'])
                
                placeholders = ', '.join(['?' for _ in result_dict])
                columns = ', '.join(result_dict.keys())
                
                conn.execute(
                    f'INSERT OR REPLACE INTO evaluations ({columns}) VALUES ({placeholders})',
                    list(result_dict.values())
                )
                
            self.logger.info(f"Stored evaluation result: {result.eval_id}")
            
        except Exception as e:
            self.logger.error(f"Error storing evaluation result: {e}")
    
    def get_evaluations(self,
                       model_name: Optional[str] = None,
                       task_type: Optional[str] = None,
                       start_time: Optional[str] = None,
                       end_time: Optional[str] = None,
                       limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve evaluation results
        
        Args:
            model_name: Filter by model name
            task_type: Filter by task type
            start_time: Start time filter (ISO format)
            end_time: End time filter (ISO format)
            limit: Maximum number of results
            
        Returns:
            List of evaluation dictionaries
        """
        query = "SELECT * FROM evaluations WHERE 1=1"
        params = []
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                results = [dict(row) for row in cursor.fetchall()]
                
                # Parse task_specific_scores JSON
                for result in results:
                    if result['task_specific_scores']:
                        result['task_specific_scores'] = json.loads(result['task_specific_scores'])
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error retrieving evaluations: {e}")
            return []
    
    def get_evaluation_summary(self,
                              model_name: Optional[str] = None,
                              task_type: Optional[str] = None,
                              start_time: Optional[str] = None,
                              end_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for evaluations
        
        Args:
            model_name: Filter by model name
            task_type: Filter by task type
            start_time: Start time filter (ISO format)
            end_time: End time filter (ISO format)
            
        Returns:
            Dictionary with evaluation summary statistics
        """
        evaluations = self.get_evaluations(
            model_name=model_name,
            task_type=task_type,
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        if not evaluations:
            return {}
        
        # Calculate summary statistics
        quality_scores = [e['quality_score'] for e in evaluations if e['quality_score'] is not None]
        relevance_scores = [e['relevance_score'] for e in evaluations if e['relevance_score'] is not None]
        coherence_scores = [e['coherence_score'] for e in evaluations if e['coherence_score'] is not None]
        accuracy_scores = [e['accuracy_score'] for e in evaluations if e['accuracy_score'] is not None]
        
        summary = {
            'total_evaluations': len(evaluations),
            'average_quality': statistics.mean(quality_scores) if quality_scores else None,
            'average_relevance': statistics.mean(relevance_scores) if relevance_scores else None,
            'average_coherence': statistics.mean(coherence_scores) if coherence_scores else None,
            'average_accuracy': statistics.mean(accuracy_scores) if accuracy_scores else None,
            'quality_distribution': self._get_score_distribution(quality_scores),
            'task_breakdown': self._get_task_breakdown(evaluations),
            'model_breakdown': self._get_model_breakdown(evaluations)
        }
        
        return summary
    
    def _get_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Get distribution of scores by ranges"""
        if not scores:
            return {}
        
        distribution = {
            '0.0-0.2': 0,
            '0.2-0.4': 0,
            '0.4-0.6': 0,
            '0.6-0.8': 0,
            '0.8-1.0': 0
        }
        
        for score in scores:
            if score < 0.2:
                distribution['0.0-0.2'] += 1
            elif score < 0.4:
                distribution['0.2-0.4'] += 1
            elif score < 0.6:
                distribution['0.4-0.6'] += 1
            elif score < 0.8:
                distribution['0.6-0.8'] += 1
            else:
                distribution['0.8-1.0'] += 1
        
        return distribution
    
    def _get_task_breakdown(self, evaluations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get breakdown of evaluations by task type"""
        task_counts = {}
        for eval_data in evaluations:
            task = eval_data['task_type']
            task_counts[task] = task_counts.get(task, 0) + 1
        return task_counts
    
    def _get_model_breakdown(self, evaluations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get breakdown of evaluations by model"""
        model_counts = {}
        for eval_data in evaluations:
            model = eval_data['model_name']
            model_counts[model] = model_counts.get(model, 0) + 1
        return model_counts