#!/usr/bin/env python3
"""
Validity Check Script for Longbow-Quarrel

Compares output correctness between longbow-quarrel and llama.cpp
using a set of factual benchmark questions.
"""

import subprocess
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from enum import Enum


class ValidityResult(Enum):
    """Validity assessment outcomes"""
    MATCH = "match"
    PARTIAL = "partial"
    MISMATCH = "mismatch"
    ERROR = "error"


@dataclass
class Question:
    """Represents a benchmark question"""
    number: int
    text: str
    category: str


@dataclass
class ComparisonResult:
    """Result of comparing two model outputs"""
    question: Question
    quarrel_output: str
    llama_output: str
    validity: ValidityResult
    similarity_score: float


class ModelRunner:
    """Handles execution of inference engines"""
    
    def __init__(self, quarrel_path: str, llama_path: str, model_path: str):
        self.quarrel_path = Path(quarrel_path)
        self.llama_path = Path(llama_path)
        self.model_path = Path(model_path)
        
        if not self.quarrel_path.exists():
            raise FileNotFoundError(f"Quarrel binary not found: {quarrel_path}")
        if not self.llama_path.exists():
            raise FileNotFoundError(f"Llama binary not found: {llama_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    def run_quarrel(self, prompt: str, n_tokens: int = 50) -> str:
        """Run longbow-quarrel and return generated text"""
        try:
            result = subprocess.run(
                [str(self.quarrel_path), "-model", str(self.model_path), 
                 "-prompt", prompt, "-n", str(n_tokens)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Extract decoded text from output
            for line in result.stderr.split('\n'):
                if 'Decoded Text:' in line:
                    return line.split('Decoded Text:')[1].strip()
            
            return result.stderr.strip()
        except subprocess.TimeoutExpired:
            return "[TIMEOUT]"
        except Exception as e:
            return f"[ERROR: {str(e)}]"
    
    def run_llama(self, prompt: str, n_tokens: int = 50) -> str:
        """Run llama.cpp and return generated text"""
        try:
            result = subprocess.run(
                ["/opt/homebrew/bin/llama-completion", "-m", str(self.model_path),
                 "-p", prompt, "-n", str(n_tokens), 
                 "--log-disable"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up llama output
            output = result.stdout.strip()
            # Remove prompt echo and headers
            lines = [l for l in output.split('\n') 
                    if not l.startswith('>') and not l.startswith('build') 
                    and not l.startswith('model')]
            print(f"DEBUG: Running Llama with prompt: {prompt[:30]}...")
            return ' '.join(lines).strip()
        except subprocess.TimeoutExpired:
            return "[TIMEOUT]"
        except Exception as e:
            return f"[ERROR: {str(e)}]"


class QuestionParser:
    """Parses benchmark questions from markdown file"""
    
    @staticmethod
    def parse(filepath: Path) -> List[Question]:
        """Parse questions from markdown file"""
        questions: List[Question] = []
        current_category = "Unknown"
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Detect category headers
                if line.startswith('##') and '(' in line:
                    current_category = line.split('##')[1].split('(')[0].strip()
                
                # Parse numbered questions
                match = re.match(r'^(\d+)\.\s+(.+)$', line)
                if match:
                    num, text = match.groups()
                    questions.append(Question(
                        number=int(num),
                        text=text,
                        category=current_category
                    ))
        
        return questions


class SimilarityEvaluator:
    """Evaluates similarity between model outputs"""
    
    @staticmethod
    def jaccard_similarity(text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    @staticmethod
    def assess_validity(similarity: float) -> ValidityResult:
        """Determine validity based on similarity score"""
        if similarity >= 0.7:
            return ValidityResult.MATCH
        elif similarity >= 0.4:
            return ValidityResult.PARTIAL
        else:
            return ValidityResult.MISMATCH


def main() -> int:
    """Main execution"""
    # Configuration
    quarrel_bin = "./quarrel"
    llama_bin = "/opt/homebrew/bin/llama-completion"
    model_path = "/Users/rsd/.ollama/models/blobs/sha256-f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57"
    questions_file = Path("docs/generic_questions.md")
    
    print("=" * 70)
    print("Longbow-Quarrel Validity Check")
    print("=" * 70)
    print()
    
    # Parse questions
    if not questions_file.exists():
        print(f"Error: Questions file not found: {questions_file}")
        return 1
    
    questions = QuestionParser.parse(questions_file)
    print(f"Loaded {len(questions)} questions")
    print()
    
    # Initialize runners
    try:
        runner = ModelRunner(quarrel_bin, llama_bin, model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Run comparisons
    results: List[ComparisonResult] = []
    
    for i, question in enumerate(questions[:20], 1):  # Test first 20 for now
        print(f"[{i}/20] {question.text}")
        
        quarrel_out = runner.run_quarrel(question.text, n_tokens=30)
        llama_out = runner.run_llama(question.text, n_tokens=30)
        
        similarity = SimilarityEvaluator.jaccard_similarity(quarrel_out, llama_out)
        validity = SimilarityEvaluator.assess_validity(similarity)
        
        results.append(ComparisonResult(
            question=question,
            quarrel_output=quarrel_out,
            llama_output=llama_out,
            validity=validity,
            similarity_score=similarity
        ))
        
        print(f"  Similarity: {similarity:.2f} ({validity.value})")
        print()
    
    # Summary statistics
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    match_count = sum(1 for r in results if r.validity == ValidityResult.MATCH)
    partial_count = sum(1 for r in results if r.validity == ValidityResult.PARTIAL)
    mismatch_count = sum(1 for r in results if r.validity == ValidityResult.MISMATCH)
    
    total = len(results)
    print(f"Match:    {match_count}/{total} ({100*match_count/total:.1f}%)")
    print(f"Partial:  {partial_count}/{total} ({100*partial_count/total:.1f}%)")
    print(f"Mismatch: {mismatch_count}/{total} ({100*mismatch_count/total:.1f}%)")
    print()
    print(f"Average Similarity: {sum(r.similarity_score for r in results)/total:.3f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
