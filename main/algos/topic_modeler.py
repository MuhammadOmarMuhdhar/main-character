#!/usr/bin/env python3
"""
BERTopic-based Topic Modeling for Bluesky Posts
Uses semantic embeddings + Gemini for high-quality topic labeling
English-only filtering for coherent results
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

try:
    import langdetect
except ImportError:
    print("Installing langdetect...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'langdetect'])
    import langdetect

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'sentence-transformers'])
    from sentence_transformers import SentenceTransformer

try:
    from bertopic import BERTopic
    from bertopic.representation import BaseRepresentation
except ImportError:
    print("Installing bertopic...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'bertopic'])
    from bertopic import BERTopic
    from bertopic.representation import BaseRepresentation

try:
    import google.generativeai as genai
except ImportError:
    print("Installing google-generativeai...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'google-generativeai'])
    import google.generativeai as genai

try:
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'scikit-learn'])
    from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class TopicResult:
    """Single topic result with embeddings"""
    id: int
    label: str
    keywords: List[str]
    post_count: int
    percentage: float
    word_scores: Dict[str, float]
    topic_embedding: List[float]  # New: topic centroid embedding


@dataclass
class TopicAnalysisResult:
    """Complete topic analysis result"""
    topics: List[TopicResult]
    metadata: Dict[str, Any]


class GeminiTopicLabeler(BaseRepresentation):
    """Custom Gemini-based topic labeling for BERTopic"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        """
        Initialize Gemini topic labeler
        
        Args:
            api_key: Google API key for Gemini
            model: Gemini model to use
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def extract_topics(self, topic_model, documents, c_tf_idf, topics, **kwargs):
        """
        Extract topic labels using Gemini with actual post content
        
        Args:
            topic_model: The BERTopic model instance  
            documents: List of documents
            c_tf_idf: Class-based TF-IDF representation
            topics: Topic IDs
            **kwargs: Additional arguments from BERTopic
            
        Returns:
            Dict mapping topic IDs to representations
        """
        topic_representations = {}
        
        # Get top words for each topic from c_tf_idf
        words = topic_model.vectorizer_model.get_feature_names_out()
        
        # Get document assignments for each topic
        topic_assignments = topic_model.topics_
        
        for topic in topics:
            if topic == -1:  # Skip outlier topic
                topic_representations[topic] = ["outlier_topic"]
                continue
            
            # Get top 10 words for this topic
            topic_words = [words[i] for i in c_tf_idf[topic].toarray().flatten().argsort()[-10:][::-1]]
            
            # Get sample posts from this topic (up to 5 representative ones)
            topic_post_indices = [i for i, t in enumerate(topic_assignments) if t == topic]
            sample_posts = []
            
            # Convert documents to list if it's a pandas object
            doc_list = documents
            if hasattr(documents, 'tolist'):
                doc_list = documents.tolist()
            elif hasattr(documents, 'values'):
                doc_list = documents.values.tolist() if hasattr(documents.values, 'tolist') else list(documents.values)
            
            # Take up to 5 posts, preferring shorter ones for better prompt efficiency
            for idx in topic_post_indices[:10]:  # Check first 10 posts in topic
                if len(sample_posts) >= 5:
                    break
                if idx < len(doc_list):  # Safety check
                    post = str(doc_list[idx]).strip()
                    if post and len(post) < 200:  # Prefer shorter posts
                        sample_posts.append(post)
            
            # If we don't have enough short posts, take some longer ones
            if len(sample_posts) < 3:
                for idx in topic_post_indices:
                    if len(sample_posts) >= 5:
                        break
                    if idx < len(doc_list):  # Safety check
                        post = str(doc_list[idx]).strip()
                        if post and post not in sample_posts:
                            # Truncate long posts
                            truncated_post = post[:150] + "..." if len(post) > 150 else post
                            sample_posts.append(truncated_post)
            
            # Generate label using Gemini with actual posts
            try:
                posts_text = "\n".join([f"- {post}" for post in sample_posts[:5]])
                
                prompt = f"""
                Create a concise, punchy topic label for controversial social media discussions. These posts are getting "ratio'd" (negative engagement) on social platforms:

                Sample posts from this topic:
                {posts_text}

                Key terms: {', '.join(topic_words[:8])}

                Guidelines:
                - 2-4 words maximum
                - Capture the specific debate/controversy 
                - Be descriptive, not generic
                - Focus on what people are arguing about
                - Examples of good labels: "Cancel Culture Debate", "AI Job Fears", "Climate Policy Fight"
                - Examples of bad labels: "Technology", "Politics", "Social Issues"

                Topic label:"""
                
                response = self.model.generate_content(prompt)
                label = response.text.strip()
                
                # Clean up the label
                if label.startswith('"') and label.endswith('"'):
                    label = label[1:-1]
                
                # Fallback to keyword-based label if Gemini fails
                if not label or len(label) > 50:
                    # Create a better fallback based on keywords
                    if len(topic_words) >= 2:
                        label = f"{topic_words[0].title()} {topic_words[1].title()}"
                    else:
                        label = topic_words[0].title() if topic_words else f"Topic {topic}"
                
                topic_representations[topic] = [label]
                
            except Exception as e:
                print(f"Gemini labeling failed for topic {topic}: {e}")
                # Better fallback based on keywords
                if len(topic_words) >= 2:
                    label = f"{topic_words[0].title()} {topic_words[1].title()}"
                else:
                    label = topic_words[0].title() if topic_words else f"Topic {topic}"
                topic_representations[topic] = [label]
        
        return topic_representations


class EnglishTopicModeler:
    """English-only topic modeling using BERTopic + Gemini"""
    
    def __init__(self, 
                 min_topic_size: int = 3,
                 max_topics: int = 15,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize English topic modeler
        
        Args:
            min_topic_size: Minimum posts per topic
            max_topics: Maximum number of topics
            embedding_model: Sentence transformer model for embeddings
        """
        self.min_topic_size = min_topic_size
        self.max_topics = max_topics
        
        # Initialize sentence transformer for embeddings
        print(f"Loading embedding model: {embedding_model}")
        self.sentence_model = SentenceTransformer(embedding_model)
        
        # Initialize Gemini (will be set up when API key is available)
        self.gemini_labeler = None
        
        # Set up BERTopic components
        self.vectorizer = CountVectorizer(
            stop_words="english",
            min_df=1,
            ngram_range=(1, 2)
        )
    
    def setup_gemini(self, api_key: str):
        """Setup Gemini labeler with API key"""
        try:
            self.gemini_labeler = GeminiTopicLabeler(api_key)
            print("✅ Gemini labeler initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            self.gemini_labeler = None
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., 'en', 'pt', etc.)
        """
        try:
            return langdetect.detect(text)
        except:
            return 'unknown'
    
    def filter_english_only(self, texts: List[str]) -> Tuple[List[str], List[int]]:
        """
        Filter texts to English only
        
        Args:
            texts: List of texts to filter
            
        Returns:
            Tuple of (english_texts, original_indices)
        """
        english_texts = []
        original_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                lang = self.detect_language(text.strip())
                if lang == 'en':
                    english_texts.append(text.strip())
                    original_indices.append(i)
        
        print(f"Filtered to {len(english_texts)} English texts from {len(texts)} total")
        return english_texts, original_indices
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate sentence embeddings for texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def extract_word_scores(self, topic_model: BERTopic, topic_id: int) -> Dict[str, float]:
        """
        Extract word scores for a topic using c-TF-IDF
        
        Args:
            topic_model: Fitted BERTopic model
            topic_id: Topic ID to extract scores for
            
        Returns:
            Dict mapping words to TF-IDF scores
        """
        try:
            # Get topic representation
            topic_words = topic_model.get_topic(topic_id)
            
            if not topic_words:
                return {}
            
            # Convert to word scores dict
            word_scores = {}
            for word, score in topic_words[:25]:  # Top 25 keywords
                word_scores[word] = round(float(score), 3)
            
            return word_scores
            
        except Exception as e:
            print(f"Error extracting word scores for topic {topic_id}: {e}")
            return {}
    
    def calculate_topic_embedding(self, embeddings: np.ndarray, labels: np.ndarray, topic_id: int) -> List[float]:
        """
        Calculate centroid embedding for a topic
        
        Args:
            embeddings: All document embeddings
            labels: Topic labels for each document
            topic_id: Topic ID to calculate centroid for
            
        Returns:
            List representing topic centroid embedding
        """
        try:
            # Get embeddings for documents in this topic
            topic_mask = labels == topic_id
            topic_embeddings = embeddings[topic_mask]
            
            if len(topic_embeddings) == 0:
                return [0.0] * embeddings.shape[1]
            
            # Calculate centroid (mean of all embeddings in topic)
            centroid = np.mean(topic_embeddings, axis=0)
            return centroid.tolist()
            
        except Exception as e:
            print(f"Error calculating topic embedding for topic {topic_id}: {e}")
            return [0.0] * embeddings.shape[1]
    
    def analyze_corpus(self, texts: List[str], api_key: Optional[str] = None) -> TopicAnalysisResult:
        """
        Perform topic modeling on English-only corpus
        
        Args:
            texts: List of texts to analyze
            api_key: Optional Gemini API key for labeling
            
        Returns:
            TopicAnalysisResult with topics and metadata
        """
        if len(texts) < 3:
            print(f"Corpus too small ({len(texts)} texts) for topic modeling")
            return TopicAnalysisResult(
                topics=[],
                metadata={
                    "error": "Corpus too small",
                    "corpus_size": len(texts),
                    "english_texts": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        try:
            # Since main character posts are already filtered for English, use all texts
            english_texts = [text.strip() for text in texts if text and text.strip()]
            
            if len(english_texts) < 3:
                print(f"Not enough texts ({len(english_texts)}) for topic modeling")
                return TopicAnalysisResult(
                    topics=[],
                    metadata={
                        "error": "Not enough texts",
                        "corpus_size": len(texts),
                        "english_texts": len(english_texts),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            
            # Generate embeddings
            embeddings = self.generate_embeddings(english_texts)
            
            # Setup Gemini if API key provided
            if api_key:
                self.setup_gemini(api_key)
            
            # Initialize BERTopic
            topic_model = BERTopic(
                embedding_model=None,  # We provide pre-computed embeddings
                vectorizer_model=self.vectorizer,
                representation_model=self.gemini_labeler,  # Use Gemini for labeling
                min_topic_size=self.min_topic_size,
                nr_topics=self.max_topics,
                calculate_probabilities=False,  # Speed up processing
                verbose=True
            )
            
            # Fit topic model with pre-computed embeddings
            print("Running BERTopic clustering...")
            topics, probabilities = topic_model.fit_transform(english_texts, embeddings)
            
            # Get topic info
            topic_info = topic_model.get_topic_info()
            
            # Extract topics
            topic_results = []
            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                
                # Skip outlier topic (-1)
                if topic_id == -1:
                    continue
                
                # Get topic label (from Gemini or fallback)
                topic_repr = topic_model.get_topic(topic_id)
                if self.gemini_labeler and topic_id in topic_model.topic_representations_:
                    label = topic_model.topic_representations_[topic_id][0]
                else:
                    # Fallback to keyword-based label
                    top_words = [word for word, _ in topic_repr[:2]]
                    label = f"{top_words[0].title()} & {top_words[1].title()}" if len(top_words) >= 2 else top_words[0].title()
                
                # Extract keywords and word scores
                if isinstance(topic_repr, list) and len(topic_repr) > 0 and isinstance(topic_repr[0], tuple):
                    keywords = [word for word, _ in topic_repr[:25]]
                else:
                    # Fallback if format is unexpected
                    keywords = [str(item) for item in topic_repr[:25]] if topic_repr else []
                word_scores = self.extract_word_scores(topic_model, topic_id)
                
                # Calculate topic embedding (centroid)
                topic_embedding = self.calculate_topic_embedding(embeddings, np.array(topics), topic_id)
                
                # Create topic result
                topic_result = TopicResult(
                    id=len(topic_results) + 1,  # Re-number starting from 1
                    label=label,
                    keywords=keywords,
                    post_count=int(row['Count']),
                    percentage=round((int(row['Count']) / len(english_texts)) * 100, 1),
                    word_scores=word_scores,
                    topic_embedding=topic_embedding
                )
                
                topic_results.append(topic_result)
            
            # Sort by post count
            topic_results.sort(key=lambda t: t.post_count, reverse=True)
            
            # Update IDs to match sorted order
            for i, topic in enumerate(topic_results):
                topic.id = i + 1
            
            # Create metadata
            metadata = {
                "total_posts": len(texts),
                "english_texts": len(english_texts),
                "topics_found": len(topic_results),
                "min_topic_size": self.min_topic_size,
                "embedding_model": self.sentence_model.get_sentence_embedding_dimension(),
                "gemini_enabled": self.gemini_labeler is not None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            print(f"✅ Topic modeling complete: {len(topic_results)} topics identified")
            return TopicAnalysisResult(topics=topic_results, metadata=metadata)
            
        except Exception as e:
            print(f"❌ Topic modeling failed: {e}")
            import traceback
            traceback.print_exc()
            
            return TopicAnalysisResult(
                topics=[],
                metadata={
                    "error": str(e),
                    "corpus_size": len(texts),
                    "english_texts": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
    
    def analyze_main_character_corpus(self, posts_data: Dict[str, Any], deep_dive_results: List[Dict[str, Any]], api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze topics in main character conversations (English only)
        
        Args:
            posts_data: Original posts data (for controversial posts)
            deep_dive_results: Deep dive results with replies/quotes
            api_key: Optional Gemini API key
            
        Returns:
            Topics analysis in JSON format for saving
        """
        print("🏷️ Starting English-only main character topic analysis...")
        
        # Build corpus from main character posts and responses
        corpus = []
        
        # Add controversial posts (from deep dive results for consistency)
        for deep_dive_result in deep_dive_results:
            original_post = deep_dive_result.get('original_post', {})
            text = original_post.get('text', '').strip()
            if text:
                corpus.append(text)
        
        # Add replies and quotes
        for deep_dive_result in deep_dive_results:
            # Add replies
            for reply in deep_dive_result.get('top_5_replies', []):
                text = reply.get('text', '').strip()
                if text:
                    corpus.append(text)
            
            # Add quotes
            for quote in deep_dive_result.get('top_5_quotes', []):
                text = quote.get('text', '').strip()
                if text:
                    corpus.append(text)
        
        print(f"Extracted {len(corpus)} texts from main character conversations")
        
        # Perform topic modeling
        result = self.analyze_corpus(corpus, api_key)
        
        # Convert to JSON format
        topics_json = {
            "collection_date": datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": result.metadata,
            "topics": []
        }
        
        for topic in result.topics:
            topics_json["topics"].append({
                "id": topic.id,
                "label": topic.label,
                "keywords": topic.keywords,
                "post_count": topic.post_count,
                "percentage": topic.percentage,
                "word_scores": topic.word_scores,
                "topic_embedding": topic.topic_embedding
            })
        
        return topics_json


def main():
    """CLI interface for BERTopic modeling"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze topics in Bluesky posts using BERTopic')
    parser.add_argument('input_file', help='JSON file with posts data')
    parser.add_argument('--output', help='Output file for topic results')
    parser.add_argument('--min-topic-size', type=int, default=3,
                       help='Minimum topic size (default: 3)')
    parser.add_argument('--max-topics', type=int, default=15,
                       help='Maximum number of topics (default: 15)')
    parser.add_argument('--gemini-key', help='Gemini API key for topic labeling')
    
    args = parser.parse_args()
    
    # Load posts data
    with open(args.input_file, 'r') as f:
        posts_data = json.load(f)
    
    # Initialize topic modeler
    modeler = EnglishTopicModeler(
        min_topic_size=args.min_topic_size,
        max_topics=args.max_topics
    )
    
    # Extract corpus (simplified for CLI)
    corpus = []
    for post in posts_data.get('posts', []):
        text = post.get('text', '').strip()
        if text:
            corpus.append(text)
    
    # Analyze topics
    result = modeler.analyze_corpus(corpus, args.gemini_key)
    
    # Print results
    print("\n📊 BERTOPIC ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Topics found: {len(result.topics)}")
    print(f"English texts analyzed: {result.metadata.get('english_texts', 0)}")
    
    for topic in result.topics:
        print(f"\n🏷️ Topic {topic.id}: {topic.label}")
        print(f"   Posts: {topic.post_count} ({topic.percentage}%)")
        print(f"   Keywords: {', '.join(topic.keywords[:10])}")
    
    # Save results if requested
    if args.output:
        topics_json = {
            "collection_date": datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": result.metadata,
            "topics": []
        }
        
        for topic in result.topics:
            topics_json["topics"].append({
                "id": topic.id,
                "label": topic.label,
                "keywords": topic.keywords,
                "post_count": topic.post_count,
                "percentage": topic.percentage,
                "word_scores": topic.word_scores,
                "topic_embedding": topic.topic_embedding
            })
        
        with open(args.output, 'w') as f:
            json.dump(topics_json, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()