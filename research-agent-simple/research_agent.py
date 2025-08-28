"""
Research Agent - AI-Powered Research with ML Analytics
Simple Python application using scikit-learn and OpenRouter API

Developed by Abdul Haseeb
LinkedIn: https://linkedin.com/in/abdul-haseeb
GitHub: https://github.com/abdul-haseeb
Email: abdul.haseeb@example.com
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# ML and Data Science imports
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot

# Web scraping and API imports
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import openai

# Load environment variables
load_dotenv()

class ResearchAgent:
    """AI-powered research agent with ML analytics"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "Research Agent")
        self.model = "meta-llama/llama-4-maverick:free"
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required in .env file")
        
        # Configure OpenAI client for OpenRouter
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name,
            }
        )
        
        # Initialize ML components
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        
        # Create outputs directory
        self.outputs_dir = Path("outputs")
        self.outputs_dir.mkdir(exist_ok=True)
        
        print("🤖 Research Agent initialized successfully!")
        print(f"📁 Output directory: {self.outputs_dir.absolute()}")
        print(f"🔑 Using model: {self.model}")
    
    def web_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search the web using DuckDuckGo"""
        print(f"🔍 Searching for: {query}")
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
            print(f"✅ Found {len(results)} search results")
            return results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def fetch_content(self, url: str, max_chars: int = 3000) -> Dict[str, Any]:
        """Fetch and extract content from a URL"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"
            }
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script and style elements
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            
            # Get main content
            main = soup.find(["article", "main"]) or soup.body or soup
            text = main.get_text(separator="\n", strip=True)
            
            # Clean up text
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)
            
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            
            title = soup.title.string.strip() if soup.title and soup.title.string else url
            
            return {
                "title": title,
                "url": url,
                "content": text,
                "success": True
            }
        except Exception as e:
            return {
                "title": url,
                "url": url,
                "content": f"Error fetching content: {str(e)}",
                "success": False
            }
    
    def conduct_research(self, query: str, max_sources: int = 5) -> Dict[str, Any]:
        """Conduct comprehensive research on a topic"""
        print(f"\n🎯 Starting research on: {query}")
        
        # Step 1: Search for sources
        search_results = self.web_search(query, max_sources)
        
        if not search_results:
            return {
                "query": query,
                "answer": "No search results found for this query.",
                "sources": [],
                "success": False
            }
        
        # Step 2: Fetch content from top sources
        print("📖 Fetching content from sources...")
        sources = []
        for i, result in enumerate(search_results[:3], 1):
            print(f"   {i}. {result['title'][:60]}...")
            content = self.fetch_content(result["url"])
            if content["success"]:
                sources.append({
                    "title": result["title"],
                    "url": result["url"],
                    "snippet": result["snippet"],
                    "content": content["content"][:1000]
                })
        
        # Step 3: Generate AI response
        print("🧠 Generating AI analysis...")
        context = self._build_context(query, sources)
        ai_response = self._generate_ai_response(context)
        
        # Step 4: Save report
        report_data = {
            "query": query,
            "answer": ai_response,
            "sources": sources,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "success": True
        }
        
        self._save_report(report_data)
        
        print("✅ Research completed successfully!")
        return report_data
    
    def _build_context(self, query: str, sources: List[Dict]) -> str:
        """Build context for AI response generation"""
        context = f"Research Query: {query}\n\n"
        context += "Sources:\n"
        
        for i, source in enumerate(sources, 1):
            context += f"\n[{i}] {source['title']}\n"
            context += f"URL: {source['url']}\n"
            context += f"Content: {source['content']}\n"
        
        return context
    
    def _generate_ai_response(self, context: str) -> str:
        """Generate AI response using OpenRouter"""
        try:
            system_prompt = """You are a meticulous research assistant. Your job is to:
1. Analyze the provided sources carefully
2. Synthesize information from multiple sources
3. Provide a comprehensive, well-structured answer
4. Include inline citations using [1], [2], etc. format
5. Be factual and cite sources appropriately
6. If information is conflicting or unclear, mention this

Format your response in markdown with proper headings and structure."""

            user_prompt = f"""Based on the following sources, provide a comprehensive research answer:

{context}

Please provide a detailed, well-researched response with proper citations."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating AI response: {str(e)}"
    
    def _save_report(self, report_data: Dict[str, Any]) -> str:
        """Save research report to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c for c in report_data["query"][:50] if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        filename = f"research_report_{timestamp}_{safe_query}.md"
        filepath = self.outputs_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Research Report\n\n")
            f.write(f"**Query:** {report_data['query']}\n")
            f.write(f"**Generated:** {report_data['timestamp']}\n\n")
            f.write("---\n\n")
            f.write(report_data["answer"])
            f.write("\n\n---\n\n## Sources\n\n")
            
            for i, source in enumerate(report_data["sources"], 1):
                f.write(f"[{i}] **{source['title']}**\n")
                f.write(f"URL: {source['url']}\n\n")
        
        print(f"💾 Report saved: {filepath}")
        return str(filepath)
    
    def analyze_similarity(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze text similarity using TF-IDF and cosine similarity"""
        print(f"\n📊 Analyzing similarity for {len(texts)} texts...")
        
        if len(texts) < 2:
            print("❌ Need at least 2 texts for similarity analysis")
            return {"error": "Need at least 2 texts", "success": False}
        
        try:
            # Vectorize texts
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Get feature names and top features
            feature_names = self.vectorizer.get_feature_names_out()
            feature_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(feature_scores)[-20:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            
            result = {
                "similarity_matrix": similarity_matrix.tolist(),
                "feature_count": len(feature_names),
                "top_features": top_features,
                "success": True
            }
            
            # Generate visualization
            self._plot_similarity_heatmap(similarity_matrix, texts)
            
            print("✅ Similarity analysis completed!")
            return result
            
        except Exception as e:
            print(f"❌ Similarity analysis failed: {e}")
            return {"error": str(e), "success": False}
    
    def cluster_texts(self, texts: List[str], n_clusters: int = 3) -> Dict[str, Any]:
        """Cluster texts using K-means clustering"""
        print(f"\n🎯 Clustering {len(texts)} texts into {n_clusters} clusters...")
        
        if len(texts) < n_clusters:
            print(f"❌ Need at least {n_clusters} texts for clustering")
            return {"error": f"Need at least {n_clusters} texts", "success": False}
        
        try:
            # Vectorize texts
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
            
            # Perform PCA for visualization
            pca = PCA(n_components=2, random_state=42)
            pca_result = pca.fit_transform(tfidf_matrix.toarray())
            
            result = {
                "cluster_labels": cluster_labels.tolist(),
                "pca_coordinates": pca_result.tolist(),
                "inertia": float(kmeans.inertia_),
                "silhouette_score": float(silhouette_avg),
                "n_clusters": n_clusters,
                "success": True
            }
            
            # Generate visualization
            self._plot_cluster_scatter(pca_result, cluster_labels, texts)
            
            print(f"✅ Clustering completed! Silhouette score: {silhouette_avg:.3f}")
            return result
            
        except Exception as e:
            print(f"❌ Clustering failed: {e}")
            return {"error": str(e), "success": False}
    
    def _plot_similarity_heatmap(self, similarity_matrix: np.ndarray, texts: List[str]):
        """Generate similarity heatmap visualization"""
        try:
            plt.figure(figsize=(10, 8))
            
            # Create labels
            labels = [f"Text {i+1}" for i in range(len(texts))]
            
            # Create heatmap
            sns.heatmap(
                similarity_matrix,
                annot=True,
                fmt='.3f',
                cmap='viridis',
                xticklabels=labels,
                yticklabels=labels,
                square=True
            )
            
            plt.title('Text Similarity Heatmap', fontsize=16, fontweight='bold')
            plt.xlabel('Documents', fontsize=12)
            plt.ylabel('Documents', fontsize=12)
            plt.tight_layout()
            
            # Save plot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"similarity_heatmap_{timestamp}.png"
            filepath = self.outputs_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Similarity heatmap saved: {filepath}")
            
        except Exception as e:
            print(f"❌ Failed to generate heatmap: {e}")
    
    def _plot_cluster_scatter(self, pca_result: np.ndarray, cluster_labels: np.ndarray, texts: List[str]):
        """Generate cluster scatter plot visualization"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot
            colors = plt.cm.Set3(np.linspace(0, 1, len(set(cluster_labels))))
            
            for i, label in enumerate(set(cluster_labels)):
                mask = cluster_labels == label
                plt.scatter(
                    pca_result[mask, 0],
                    pca_result[mask, 1],
                    c=[colors[label]],
                    label=f'Cluster {label}',
                    alpha=0.7,
                    s=100
                )
            
            plt.title('Document Clustering (PCA Visualization)', fontsize=16, fontweight='bold')
            plt.xlabel('First Principal Component', fontsize=12)
            plt.ylabel('Second Principal Component', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"cluster_scatter_{timestamp}.png"
            filepath = self.outputs_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Cluster scatter plot saved: {filepath}")
            
        except Exception as e:
            print(f"❌ Failed to generate scatter plot: {e}")
    
    def interactive_mode(self):
        """Run the agent in interactive mode"""
        print("\n" + "="*60)
        print("🤖 RESEARCH AGENT - Interactive Mode")
        print("Developed by Abdul Haseeb")
        print("="*60)
        
        while True:
            print("\nChoose an option:")
            print("1. 🔍 Conduct Research")
            print("2. 📊 Analyze Text Similarity")
            print("3. 🎯 Cluster Texts")
            print("4. 📁 View Output Directory")
            print("5. ❌ Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                self._interactive_research()
            elif choice == "2":
                self._interactive_similarity()
            elif choice == "3":
                self._interactive_clustering()
            elif choice == "4":
                self._show_outputs()
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def _interactive_research(self):
        """Interactive research mode"""
        query = input("\n🔍 Enter your research question: ").strip()
        if not query:
            print("❌ Please enter a valid question.")
            return
        
        max_sources = input("📚 Number of sources (default 5): ").strip()
        max_sources = int(max_sources) if max_sources.isdigit() else 5
        
        result = self.conduct_research(query, max_sources)
        
        if result["success"]:
            print("\n" + "="*60)
            print("📋 RESEARCH RESULTS")
            print("="*60)
            print(result["answer"])
            print("\n" + "="*60)
        else:
            print(f"❌ Research failed: {result.get('answer', 'Unknown error')}")
    
    def _interactive_similarity(self):
        """Interactive similarity analysis mode"""
        print("\n📊 TEXT SIMILARITY ANALYSIS")
        print("Enter texts to compare (minimum 2, press Enter twice to finish):")
        
        texts = []
        text_num = 1
        
        while True:
            text = input(f"Text {text_num}: ").strip()
            if not text:
                if len(texts) >= 2:
                    break
                else:
                    print("❌ Please enter at least 2 texts.")
                    continue
            texts.append(text)
            text_num += 1
        
        result = self.analyze_similarity(texts)
        
        if result["success"]:
            print(f"\n📈 Similarity Analysis Results:")
            print(f"   • Feature count: {result['feature_count']}")
            print(f"   • Top features: {', '.join(result['top_features'][:10])}")
            
            # Show similarity matrix
            matrix = np.array(result["similarity_matrix"])
            print(f"\n📊 Similarity Matrix:")
            for i in range(len(texts)):
                for j in range(len(texts)):
                    print(f"   Text {i+1} ↔ Text {j+1}: {matrix[i,j]:.3f}")
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
    
    def _interactive_clustering(self):
        """Interactive clustering mode"""
        print("\n🎯 TEXT CLUSTERING ANALYSIS")
        print("Enter texts to cluster (minimum 3, press Enter twice to finish):")
        
        texts = []
        text_num = 1
        
        while True:
            text = input(f"Text {text_num}: ").strip()
            if not text:
                if len(texts) >= 3:
                    break
                else:
                    print("❌ Please enter at least 3 texts.")
                    continue
            texts.append(text)
            text_num += 1
        
        n_clusters = input(f"\n🎯 Number of clusters (default 3): ").strip()
        n_clusters = int(n_clusters) if n_clusters.isdigit() else 3
        
        if n_clusters > len(texts):
            n_clusters = len(texts) - 1
            print(f"⚠️  Adjusted clusters to {n_clusters} (max for {len(texts)} texts)")
        
        result = self.cluster_texts(texts, n_clusters)
        
        if result["success"]:
            print(f"\n🎯 Clustering Results:")
            print(f"   • Number of clusters: {result['n_clusters']}")
            print(f"   • Silhouette score: {result['silhouette_score']:.3f}")
            print(f"   • Inertia: {result['inertia']:.3f}")
            
            # Show cluster assignments
            labels = result["cluster_labels"]
            print(f"\n📋 Cluster Assignments:")
            for i, label in enumerate(labels):
                print(f"   Text {i+1}: Cluster {label}")
        else:
            print(f"❌ Clustering failed: {result.get('error', 'Unknown error')}")
    
    def _show_outputs(self):
        """Show contents of output directory"""
        print(f"\n📁 Output Directory: {self.outputs_dir.absolute()}")
        
        files = list(self.outputs_dir.glob("*"))
        if not files:
            print("   (Empty)")
            return
        
        print(f"\n📋 Files ({len(files)} total):")
        for file in sorted(files):
            size = file.stat().st_size
            modified = time.ctime(file.stat().st_mtime)
            print(f"   • {file.name} ({size} bytes, {modified})")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Research Agent - AI-Powered Research with ML Analytics")
    parser.add_argument("--query", "-q", help="Research query")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--similarity", "-s", nargs="+", help="Analyze similarity of provided texts")
    parser.add_argument("--cluster", "-c", nargs="+", help="Cluster provided texts")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters (default: 3)")
    
    args = parser.parse_args()
    
    try:
        agent = ResearchAgent()
        
        if args.interactive or not any([args.query, args.similarity, args.cluster]):
            agent.interactive_mode()
        elif args.query:
            result = agent.conduct_research(args.query)
            if result["success"]:
                print("\n" + "="*60)
                print("📋 RESEARCH RESULTS")
                print("="*60)
                print(result["answer"])
        elif args.similarity:
            if len(args.similarity) < 2:
                print("❌ Need at least 2 texts for similarity analysis")
                return
            result = agent.analyze_similarity(args.similarity)
            if result["success"]:
                matrix = np.array(result["similarity_matrix"])
                print(f"\n📊 Similarity Matrix:")
                for i in range(len(args.similarity)):
                    for j in range(len(args.similarity)):
                        print(f"Text {i+1} ↔ Text {j+1}: {matrix[i,j]:.3f}")
        elif args.cluster:
            if len(args.cluster) < args.clusters:
                print(f"❌ Need at least {args.clusters} texts for clustering")
                return
            result = agent.cluster_texts(args.cluster, args.clusters)
            if result["success"]:
                labels = result["cluster_labels"]
                print(f"\n🎯 Cluster Assignments:")
                for i, label in enumerate(labels):
                    print(f"Text {i+1}: Cluster {label}")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

