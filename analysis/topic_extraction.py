import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


# Define major topic categories with their keywords
TOPIC_KEYWORDS = {
    "Large Language Models": {
        "keywords": ["large language model", "llm", "language model", "gpt", "bert", "transformer", "prompt", "in-context learning", "instruction tuning"],
        "subtopics": {
            "Reasoning": ["reasoning", "llm reasoning", "chain-of-thought", "cot", "mathematical reasoning"],
            "Safety & Alignment": ["alignment", "ai safety", "safety", "jailbreak", "adversarial prompts", "harmful"],
            "Efficiency": ["quantization", "pruning", "distillation", "efficiency", "compression", "low-rank"],
            "Multimodal": ["multimodal", "vision-language", "mllm", "vlm", "image-text"],
            "Agents": ["agent", "tool use", "planning", "reasoning agent"],
            "Training & Fine-tuning": ["instruction tuning", "rlhf", "fine-tuning", "supervised fine-tuning", "sft"],
            "Evaluation": ["evaluation", "benchmark", "dataset", "metric"],
        }
    },
    "Computer Vision": {
        "keywords": ["computer vision", "image", "visual", "video", "3d", "object detection", "segmentation", "recognition"],
        "subtopics": {
            "3D Vision": ["3d", "point cloud", "mesh", "reconstruction", "3d generation", "neural radiance"],
            "Image Generation": ["image generation", "text-to-image", "image synthesis", "gan", "stable diffusion"],
            "Video": ["video", "video generation", "video understanding", "temporal", "action recognition"],
            "Object Detection": ["object detection", "detection", "yolo", "detection", "instance segmentation"],
            "Scene Understanding": ["scene", "scene understanding", "semantic segmentation", "depth estimation"],
        }
    },
    "Reinforcement Learning": {
        "keywords": ["reinforcement learning", "rl", "policy", "reward", "q-learning", "actor-critic", "ppo", "dqn"],
        "subtopics": {
            "Robotics": ["robotics", "robot", "manipulation", "navigation", "control", "embodied"],
            "Multi-Agent": ["multi-agent", "marl", "cooperative", "communication"],
            "Offline RL": ["offline", "offline rl", "batch rl", "dataset"],
            "Model-Based RL": ["model-based", "world model", "planning", "mbrl"],
            "Exploration": ["exploration", "curiosity", "intrinsic motivation"],
        }
    },
    "Generative Models": {
        "keywords": ["diffusion", "diffusion model", "generative", "gan", "vae", "flow matching", "score-based"],
        "subtopics": {
            "Diffusion Models": ["diffusion", "diffusion model", "score-based", "denoising"],
            "Flow Matching": ["flow matching", "flow", "continuous normalizing flow", "cnf"],
            "GANs": ["gan", "generative adversarial", "stylegan"],
            "VAEs": ["vae", "variational autoencoder", "latent variable"],
        }
    },
    "Graph Neural Networks": {
        "keywords": ["graph", "graph neural network", "gnn", "node", "edge", "molecular", "molecule"],
        "subtopics": {
            "Molecular": ["molecular", "molecule", "drug", "protein", "chemistry"],
            "Social Networks": ["social", "social network", "community"],
            "Knowledge Graphs": ["knowledge graph", "knowledge base", "entity", "relation"],
        }
    },
    "Interpretability & Explainability": {
        "keywords": ["interpretability", "explainability", "xai", "mechanistic interpretability", "attention", "saliency"],
        "subtopics": {
            "Mechanistic Interpretability": ["mechanistic interpretability", "circuit", "feature", "neuron"],
            "Attention Analysis": ["attention", "attention map", "attention weight"],
            "Attribution Methods": ["attribution", "saliency", "gradient", "integrated gradients"],
        }
    },
    "Optimization & Training": {
        "keywords": ["optimization", "optimizer", "gradient", "convergence", "learning rate", "adam", "sgd"],
        "subtopics": {
            "Optimizers": ["optimizer", "adam", "sgd", "momentum", "learning rate"],
            "Convergence": ["convergence", "convergence rate", "stability"],
            "Distributed Training": ["distributed", "parallel", "multi-gpu", "data parallel"],
        }
    },
    "Representation Learning": {
        "keywords": ["representation learning", "embedding", "contrastive learning", "self-supervised", "metric learning"],
        "subtopics": {
            "Contrastive Learning": ["contrastive", "contrastive learning", "simclr", "moco"],
            "Self-Supervised": ["self-supervised", "ssl", "pretraining", "masked"],
            "Metric Learning": ["metric learning", "triplet loss", "similarity"],
        }
    },
    "Robustness & Security": {
        "keywords": ["robustness", "adversarial", "attack", "defense", "security", "backdoor", "poisoning"],
        "subtopics": {
            "Adversarial Robustness": ["adversarial", "adversarial attack", "adversarial training"],
            "Backdoor Attacks": ["backdoor", "trojan", "poisoning"],
            "Certified Defense": ["certified", "provable", "verification"],
        }
    },
    "Natural Language Processing": {
        "keywords": ["nlp", "natural language", "text", "language", "translation", "summarization", "sentiment"],
        "subtopics": {
            "Machine Translation": ["translation", "machine translation", "nmt"],
            "Summarization": ["summarization", "summarize", "abstractive"],
            "Information Extraction": ["information extraction", "named entity", "relation extraction"],
            "Question Answering": ["question answering", "qa", "reading comprehension"],
        }
    },
    "Federated & Distributed Learning": {
        "keywords": ["federated", "federated learning", "distributed", "privacy", "differential privacy"],
        "subtopics": {
            "Federated Learning": ["federated", "federated learning", "fl"],
            "Privacy": ["privacy", "differential privacy", "secure aggregation"],
        }
    },
    "Continual Learning": {
        "keywords": ["continual", "continual learning", "lifelong", "catastrophic forgetting", "incremental"],
        "subtopics": {
            "Catastrophic Forgetting": ["catastrophic forgetting", "forgetting"],
            "Task-Incremental": ["task-incremental", "incremental"],
        }
    },
    "Time Series": {
        "keywords": ["time series", "time-series", "forecasting", "temporal", "sequence"],
        "subtopics": {
            "Forecasting": ["forecasting", "prediction", "time series forecast"],
            "Anomaly Detection": ["anomaly", "anomaly detection", "outlier"],
        }
    },
}


def assign_topics_to_paper(keywords: List[str], title: str) -> Dict[str, List[str]]:
    """Assign main topics and subtopics to a paper based on keywords and title."""
    text = " ".join([k.lower() for k in keywords] + [title.lower()])
    
    assigned = defaultdict(list)
    
    for topic, config in TOPIC_KEYWORDS.items():
        # Check if main topic matches
        topic_score = sum(1 for kw in config["keywords"] if kw in text)
        
        if topic_score > 0:
            # Check subtopics
            for subtopic, sub_keywords in config["subtopics"].items():
                subtopic_score = sum(1 for kw in sub_keywords if kw in text)
                if subtopic_score > 0:
                    assigned[topic].append(subtopic)
            
            # If topic matches but no subtopic, add "General"
            if topic not in assigned or len(assigned[topic]) == 0:
                assigned[topic].append("General")
    
    return dict(assigned)


def run() -> None:
    parser = argparse.ArgumentParser(description="Extract topics from papers.")
    parser.add_argument("--input", default="data/papers.parquet")
    parser.add_argument("--out-dir", default="analysis")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    
    # Assign topics to each paper
    paper_topics = []
    topic_counts = Counter()
    topic_subtopic_counts = defaultdict(Counter)
    
    for idx, row in df.iterrows():
        keywords = []
        if row.get("keywords") is not None:
            if hasattr(row["keywords"], '__iter__') and not isinstance(row["keywords"], str):
                keywords = [str(k) for k in row["keywords"]]
            elif isinstance(row["keywords"], str):
                keywords = [row["keywords"]]
        
        title = str(row.get("title", ""))
        topics = assign_topics_to_paper(keywords, title)
        
        paper_topics.append({
            "paper_id": row["paper_id"],
            "topics": topics
        })
        
        for topic, subtopics in topics.items():
            topic_counts[topic] += 1
            for subtopic in subtopics:
                topic_subtopic_counts[topic][subtopic] += 1
    
    # Build topic hierarchy
    topic_hierarchy = {}
    for topic in sorted(topic_counts.keys()):
        topic_hierarchy[topic] = {
            "count": topic_counts[topic],
            "subtopics": {
                subtopic: count 
                for subtopic, count in sorted(
                    topic_subtopic_counts[topic].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            }
        }
    
    # Save results
    with (out_dir / "paper_topics.json").open("w", encoding="utf-8") as f:
        json.dump(paper_topics, f, indent=2, ensure_ascii=True)
    
    with (out_dir / "topic_hierarchy.json").open("w", encoding="utf-8") as f:
        json.dump(topic_hierarchy, f, indent=2, ensure_ascii=True)
    
    print(f"Extracted topics for {len(df)} papers")
    print(f"\nTopic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {topic}: {count} papers")


if __name__ == "__main__":
    run()
