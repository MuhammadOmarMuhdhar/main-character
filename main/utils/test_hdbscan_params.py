#!/usr/bin/env python3
"""
HDBSCAN Parameter Testing Script for Topic Modeling
Tests different HDBSCAN parameter combinations to find optimal clustering settings
"""

import os
import sys
import json
import numpy as np
from itertools import product
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    import hdbscan
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)


@dataclass
class TestResult:
    """Single parameter test result"""
    min_cluster_size: int
    min_samples: int
    cluster_selection_epsilon: float
    cluster_selection_method: str
    num_topics: int
    silhouette_score: float
    topic_distribution: List[int]
    sample_topics: List[Dict[str, Any]]


class HDBSCANParameterTester:
    """Test different HDBSCAN parameters for optimal topic modeling"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize tester with embedding model"""
        print(f"Loading embedding model: {embedding_model}")
        self.sentence_model = SentenceTransformer(embedding_model)
        self.vectorizer = CountVectorizer(
            stop_words="english",
            min_df=1,
            ngram_range=(1, 2)
        )
    
    def load_test_corpus(self, data_file: str) -> List[str]:
        """
        Load test corpus from main character data
        
        Args:
            data_file: Path to today.json file
            
        Returns:
            List of text strings for topic modeling
        """
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            corpus = []
            
            # Extract main character posts
            for character in data.get('main_characters', []):
                post_text = character.get('post', {}).get('text', '').strip()
                if post_text:
                    corpus.append(post_text)
                
                # Add sample replies (first 3 to avoid overwhelming)
                for reply in character.get('sample_replies', [])[:3]:
                    if reply and reply.strip():
                        corpus.append(reply.strip())
            
            print(f"Loaded {len(corpus)} texts from {data_file}")
            return corpus
            
        except Exception as e:
            print(f"Error loading test corpus: {e}")
            return []
    
    def calculate_silhouette_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate silhouette score for clustering quality
        
        Args:
            embeddings: Document embeddings
            labels: Cluster labels
            
        Returns:
            Silhouette score (higher = better clustering)
        """
        try:
            from sklearn.metrics import silhouette_score
            
            # Remove outliers (-1 labels) for scoring
            mask = labels != -1
            if np.sum(mask) < 2:
                return -1.0  # Not enough clustered points
            
            score = silhouette_score(embeddings[mask], labels[mask])
            return round(score, 3)
            
        except Exception as e:
            print(f"Error calculating silhouette score: {e}")
            return -1.0
    
    def test_parameter_combination(self, 
                                 texts: List[str], 
                                 embeddings: np.ndarray,
                                 min_cluster_size: int,
                                 cluster_selection_epsilon: float,
                                 cluster_selection_method: str,
                                 min_samples: int = None) -> TestResult:
        """
        Test a single parameter combination
        
        Args:
            texts: Text corpus
            embeddings: Pre-computed embeddings
            min_cluster_size: HDBSCAN min_cluster_size parameter
            cluster_selection_epsilon: HDBSCAN cluster_selection_epsilon parameter  
            cluster_selection_method: HDBSCAN cluster_selection_method parameter
            
        Returns:
            TestResult with metrics and sample topics
        """
        try:
            # Create custom HDBSCAN model
            hdbscan_params = {
                'min_cluster_size': min_cluster_size,
                'cluster_selection_epsilon': cluster_selection_epsilon,
                'cluster_selection_method': cluster_selection_method,
                'metric': 'euclidean'
            }
            
            # Add min_samples if provided
            if min_samples is not None:
                hdbscan_params['min_samples'] = min_samples
                
            hdbscan_model = hdbscan.HDBSCAN(**hdbscan_params)
            
            # Create BERTopic with custom HDBSCAN
            topic_model = BERTopic(
                embedding_model=None,  # Use pre-computed embeddings
                hdbscan_model=hdbscan_model,
                vectorizer_model=self.vectorizer,
                min_topic_size=2,  # Low threshold for testing
                calculate_probabilities=False,
                verbose=False
            )
            
            # Fit model
            topics, probabilities = topic_model.fit_transform(texts, embeddings)
            
            # Get topic info
            topic_info = topic_model.get_topic_info()
            
            # Calculate metrics
            unique_topics = [t for t in np.unique(topics) if t != -1]
            num_topics = len(unique_topics)
            silhouette = float(self.calculate_silhouette_score(embeddings, np.array(topics)))
            
            # Get topic distribution (posts per topic)
            topic_distribution = []
            for topic_id in unique_topics:
                count = int(np.sum(np.array(topics) == topic_id))
                topic_distribution.append(count)
            
            # Get sample topics with keywords
            sample_topics = []
            for topic_id in unique_topics[:3]:  # First 3 topics
                topic_words = topic_model.get_topic(topic_id)
                if topic_words:
                    keywords = [word for word, _ in topic_words[:5]]
                    sample_topics.append({
                        'id': int(topic_id),
                        'keywords': keywords,
                        'post_count': int(np.sum(np.array(topics) == topic_id))
                    })
            
            return TestResult(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples or 1,
                cluster_selection_epsilon=cluster_selection_epsilon,
                cluster_selection_method=cluster_selection_method,
                num_topics=num_topics,
                silhouette_score=silhouette,
                topic_distribution=topic_distribution,
                sample_topics=sample_topics
            )
            
        except Exception as e:
            print(f"Error testing parameters {min_cluster_size}, {cluster_selection_epsilon}, {cluster_selection_method}, {min_samples}: {e}")
            return TestResult(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples or 1,
                cluster_selection_epsilon=cluster_selection_epsilon,
                cluster_selection_method=cluster_selection_method,
                num_topics=0,
                silhouette_score=-1.0,
                topic_distribution=[],
                sample_topics=[]
            )
    
    def run_parameter_sweep(self, texts: List[str]) -> List[TestResult]:
        """
        Run comprehensive parameter sweep
        
        Args:
            texts: Text corpus for testing
            
        Returns:
            List of TestResult objects
        """
        if len(texts) < 10:
            print(f"Corpus too small ({len(texts)} texts) for parameter testing")
            return []
        
        print("Generating embeddings...")
        embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
        
        # Parameter ranges to test (focused for speed)
        min_cluster_sizes = [3, 4, 5, 6, 8]
        cluster_selection_epsilons = [0.0, 0.02, 0.05, 0.1, 0.15]
        cluster_selection_methods = ['eom', 'leaf']
        min_samples_values = [1, 2, 3]  # New parameter
        
        # Generate all combinations
        param_combinations = list(product(
            min_cluster_sizes,
            cluster_selection_epsilons, 
            cluster_selection_methods,
            min_samples_values
        ))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        results = []
        for i, (min_size, epsilon, method, min_samples) in enumerate(param_combinations):
            print(f"Testing {i+1}/{len(param_combinations)}: "
                  f"min_size={min_size}, epsilon={epsilon}, method={method}, min_samples={min_samples}")
            
            result = self.test_parameter_combination(
                texts, embeddings, min_size, epsilon, method, min_samples
            )
            results.append(result)
        
        return results
    
    def analyze_results(self, results: List[TestResult]) -> Dict[str, Any]:
        """
        Analyze test results and recommend best parameters
        
        Args:
            results: List of test results
            
        Returns:
            Analysis summary with recommendations
        """
        if not results:
            return {"error": "No results to analyze"}
        
        # Filter valid results
        valid_results = [r for r in results if r.num_topics > 0 and r.silhouette_score > -1]
        
        if not valid_results:
            return {"error": "No valid clustering results found"}
        
        # Find best by different criteria
        best_silhouette = max(valid_results, key=lambda r: r.silhouette_score)
        best_topic_count = min(valid_results, key=lambda r: abs(r.num_topics - 3))  # Aim for ~3 topics
        
        # Look for results with good balance (decent silhouette + reasonable topic count)
        balanced_results = [
            r for r in valid_results 
            if 2 <= r.num_topics <= 5 and r.silhouette_score > 0.1
        ]
        
        recommended = best_silhouette
        if balanced_results:
            recommended = max(balanced_results, key=lambda r: r.silhouette_score)
        
        return {
            "total_tests": len(results),
            "valid_results": len(valid_results),
            "best_silhouette": {
                "params": {
                    "min_cluster_size": best_silhouette.min_cluster_size,
                    "min_samples": best_silhouette.min_samples,
                    "cluster_selection_epsilon": best_silhouette.cluster_selection_epsilon,
                    "cluster_selection_method": best_silhouette.cluster_selection_method
                },
                "metrics": {
                    "num_topics": best_silhouette.num_topics,
                    "silhouette_score": best_silhouette.silhouette_score,
                    "topic_distribution": best_silhouette.topic_distribution
                },
                "sample_topics": best_silhouette.sample_topics
            },
            "recommended": {
                "params": {
                    "min_cluster_size": recommended.min_cluster_size,
                    "min_samples": recommended.min_samples,
                    "cluster_selection_epsilon": recommended.cluster_selection_epsilon,
                    "cluster_selection_method": recommended.cluster_selection_method
                },
                "metrics": {
                    "num_topics": recommended.num_topics,
                    "silhouette_score": recommended.silhouette_score,
                    "topic_distribution": recommended.topic_distribution
                },
                "sample_topics": recommended.sample_topics,
                "reason": "Best balance of clustering quality and topic count"
            },
            "all_results": [
                {
                    "params": {
                        "min_cluster_size": r.min_cluster_size,
                        "min_samples": r.min_samples,
                        "cluster_selection_epsilon": r.cluster_selection_epsilon,
                        "cluster_selection_method": r.cluster_selection_method
                    },
                    "metrics": {
                        "num_topics": r.num_topics,
                        "silhouette_score": r.silhouette_score,
                        "topic_distribution": r.topic_distribution
                    }
                }
                for r in valid_results
            ]
        }


def main():
    """CLI interface for HDBSCAN parameter testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test HDBSCAN parameters for topic modeling')
    parser.add_argument('--data-file', 
                       default='frontend/today.json',
                       help='Path to data file (default: frontend/today.json)')
    parser.add_argument('--output', 
                       default='main/utils/hdbscan_test_results.json',
                       help='Output file for test results')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = HDBSCANParameterTester()
    
    # Load test corpus
    corpus = tester.load_test_corpus(args.data_file)
    if not corpus:
        print("❌ Failed to load test corpus")
        return
    
    # Run parameter sweep
    print("🔬 Starting HDBSCAN parameter testing...")
    results = tester.run_parameter_sweep(corpus)
    
    if not results:
        print("❌ No test results generated")
        return
    
    # Analyze results
    print("📊 Analyzing results...")
    analysis = tester.analyze_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 HDBSCAN PARAMETER TEST RESULTS")
    print("="*60)
    
    if "error" in analysis:
        print(f"❌ {analysis['error']}")
        return
    
    print(f"Total tests run: {analysis['total_tests']}")
    print(f"Valid results: {analysis['valid_results']}")
    
    recommended = analysis['recommended']
    print(f"\n🏆 RECOMMENDED PARAMETERS:")
    print(f"  min_cluster_size: {recommended['params']['min_cluster_size']}")
    print(f"  min_samples: {recommended['params']['min_samples']}")
    print(f"  cluster_selection_epsilon: {recommended['params']['cluster_selection_epsilon']}")  
    print(f"  cluster_selection_method: {recommended['params']['cluster_selection_method']}")
    print(f"  → Results: {recommended['metrics']['num_topics']} topics, "
          f"silhouette={recommended['metrics']['silhouette_score']}")
    print(f"  → Reason: {recommended['reason']}")
    
    if recommended['sample_topics']:
        print(f"\n📝 Sample Topics:")
        for topic in recommended['sample_topics']:
            keywords = ', '.join(topic['keywords'])
            print(f"  Topic {topic['id']}: {keywords} ({topic['post_count']} posts)")
    
    # Save detailed results
    analysis['timestamp'] = datetime.now().isoformat()
    analysis['test_corpus_size'] = len(corpus)
    
    with open(args.output, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n💾 Detailed results saved to {args.output}")
    print("✅ Parameter testing complete!")


if __name__ == "__main__":
    main()