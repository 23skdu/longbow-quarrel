#!/usr/bin/env python3
"""
25 Questions Coherence and Parity Verification Suite
Compares longbow-quarrel output against llama.cpp (llama-cli)
across cached models in ~/.cache/llmfit/models/
"""

import os
import sys
import subprocess
import json
import time
import re

QUESTIONS = [
    # General Knowledge / Trivia
    "The capital of France is",
    "What is the capital of Japan?",
    "The largest planet in our solar system is",
    "Who painted the Mona Lisa?",
    "The chemical symbol for gold is",
    
    # Science & Nature
    "Water boils at a temperature of",
    "Why is the sky blue?",
    "Photosynthesis is the process by which",
    "The powerhouse of the cell is the",
    "What causes ocean tides on Earth?",
    
    # Mathematics & Logic
    "What is 15 multiplied by 4?",
    "If a triangle has angles of 90 degrees and 45 degrees, what is the third angle?",
    "Solve for x: 2x + 10 = 20. The value of x is",
    "What is the square root of 144?",
    "If all cats are mammals and all mammals breathe air, then all cats",
    
    # Code & Technology
    "In Python, the function used to get the length of a list is",
    "In SQL, the statement used to query data from a table is",
    "What does HTTP status code 404 indicate?",
    "A boolean data type can have two possible values:",
    "In Git, the command used to save local changes to the staging area is",
    
    # Language & Grammar
    "An antonym for the word 'ancient' is",
    "A synonym for 'rapid' is",
    "Translate the word 'Hello' into Spanish:",
    "The plural form of 'child' is",
    "Complete the idiom: A blessing in",
]

MODELS = [
    {
        "name": "Qwen3.5-2B-Q8_0",
        "path": "/home/rsd/.cache/llmfit/models/Qwen3.5-2B-Q8_0.gguf",
        "supported": True,
    },
    {
        "name": "Huihui-Qwen3.5-4B-Opus",
        "path": "/home/rsd/.cache/llmfit/models/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated.Q8_0.gguf",
        "supported": True,
    },
    {
        "name": "Gemma4-E2B",
        "path": "/home/rsd/.cache/llmfit/models/Gemma4_E2B_Abliterated_Opus_Distilled.Q8_0.gguf",
        "supported": False,
        "note": "Gemma 4 architecture uses hybrid sliding window + shared KV layers (IQ4_NL), not yet implemented on CPU engine",
    },
]

LLAMA_CLI = "/home/linuxbrew/.linuxbrew/bin/llama-cli"
QUARREL_CMD = ["./bin/quarrel-simple"]

def run_llama(model_path, prompt, n_tokens=25):
    cmd = [
        LLAMA_CLI,
        "--simple-io",
        "-st",
        "--reasoning", "off",
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_tokens),
        "--temp", "0",
        "--seed", "42"
    ]
    try:
        start = time.time()
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        elapsed = time.time() - start
        output = res.stdout
        # Extract text after prompt
        # In simple-io, prompt is displayed after "> "
        pattern = r">\s*" + re.escape(prompt) + r"\s*\n?(.*?)(?:\[\s*Prompt:|\Z)"
        m = re.search(pattern, output, re.DOTALL)
        if m:
            gen = m.group(1).strip()
        else:
            lines = output.splitlines()
            gen_lines = []
            found_prompt = False
            for line in lines:
                if prompt in line:
                    found_prompt = True
                    continue
                if found_prompt:
                    if "[ Prompt:" in line or "Exiting..." in line:
                        break
                    gen_lines.append(line)
            gen = "\n".join(gen_lines).strip()
        return {
            "success": res.returncode == 0,
            "text": gen,
            "raw": output,
            "elapsed": elapsed
        }
    except Exception as e:
        return {"success": False, "error": str(e), "text": ""}

def run_quarrel(model_path, prompt, n_tokens=25):
    cmd = QUARREL_CMD + [
        "--model", model_path,
        "--prompt", prompt,
        "-n", str(n_tokens),
        "--temp", "0",
        "--seed", "42"
    ]
    try:
        start = time.time()
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90)
        elapsed = time.time() - start
        output = res.stdout
        # Extract "Full output:\n..."
        m = re.search(r"Full output:\s*\n(.*)", output, re.DOTALL)
        if m:
            full = m.group(1).strip()
            # If prompt is at start, strip it to get just generated text
            if full.startswith(prompt):
                gen = full[len(prompt):].strip()
            else:
                gen = full
        else:
            # Try to find "Generated: ..." line
            m_gen = re.search(r"Generated:\s*(.*)", output)
            if m_gen:
                gen = m_gen.group(1).strip()
            else:
                gen = output.strip()
        return {
            "success": res.returncode == 0,
            "text": gen,
            "raw": output,
            "elapsed": elapsed
        }
    except Exception as e:
        return {"success": False, "error": str(e), "text": ""}

def evaluate_coherence(text):
    if not text or len(text.strip()) == 0:
        return False, "empty output"
    
    # Check for excessive repetition (e.g. same token repeated 10 times)
    words = text.split()
    if len(words) > 5:
        from collections import Counter
        counts = Counter(words)
        most_common, max_freq = counts.most_common(1)[0]
        if max_freq / len(words) > 0.6:
            return False, f"repetition loop ({most_common})"
    
    # Check ASCII / readable character ratio
    printable = sum(1 for c in text if c.isprintable() or c in '\n\r\t')
    if printable / max(1, len(text)) < 0.85:
        return False, "gibberish non-printable characters"
    
    return True, "coherent"

def compute_similarity(text1, text2):
    w1 = set(re.findall(r'\w+', text1.lower()))
    w2 = set(re.findall(r'\w+', text2.lower()))
    if not w1 or not w2:
        return 0.0
    intersection = w1.intersection(w2)
    union = w1.union(w2)
    return len(intersection) / len(union)

def main():
    results = {}
    print(f"Starting 25 Questions Coherence Verification Suite...")
    print(f"Total Questions: {len(QUESTIONS)}")
    
    for m in MODELS:
        model_name = m["name"]
        model_path = m["path"]
        is_supported = m["supported"]
        
        print(f"\n=======================================================")
        print(f"Testing Model: {model_name} (Supported: {is_supported})")
        print(f"Path: {model_path}")
        print(f"=======================================================\n")
        
        results[model_name] = {
            "supported": is_supported,
            "note": m.get("note", ""),
            "questions": []
        }
        
        if not is_supported:
            print(f"Skipping full 25 questions for {model_name}: {m.get('note')}")
            # Run single sample to record architecture behavior
            sample_q = QUESTIONS[0]
            q_res = run_quarrel(model_path, sample_q, n_tokens=10)
            l_res = run_llama(model_path, sample_q, n_tokens=10)
            results[model_name]["sample"] = {
                "question": sample_q,
                "quarrel": q_res.get("text", ""),
                "llama": l_res.get("text", "")
            }
            continue
            
        for idx, q in enumerate(QUESTIONS, 1):
            print(f"[{idx:02d}/25] Prompt: {q!r}")
            q_res = run_quarrel(model_path, q, n_tokens=25)
            l_res = run_llama(model_path, q, n_tokens=25)
            
            q_coherent, q_reason = evaluate_coherence(q_res.get("text", ""))
            l_coherent, l_reason = evaluate_coherence(l_res.get("text", ""))
            
            sim = compute_similarity(q_res.get("text", ""), l_res.get("text", ""))
            
            print(f"       Quarrel: {q_res.get('text', '')[:60]!r} (Coherent: {q_coherent})")
            print(f"       Llama  : {l_res.get('text', '')[:60]!r} (Coherent: {l_coherent})")
            print(f"       Similarity: {sim:.2f} | Quarrel Speed: {q_res.get('elapsed', 0):.1f}s | Llama Speed: {l_res.get('elapsed', 0):.1f}s")
            
            results[model_name]["questions"].append({
                "index": idx,
                "prompt": q,
                "quarrel_text": q_res.get("text", ""),
                "quarrel_coherent": q_coherent,
                "quarrel_elapsed": q_res.get("elapsed", 0),
                "llama_text": l_res.get("text", ""),
                "llama_coherent": l_coherent,
                "llama_elapsed": l_res.get("elapsed", 0),
                "similarity": sim
            })

    output_json = "coherence_verification_results.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll test results saved to {output_json}")

if __name__ == "__main__":
    main()
