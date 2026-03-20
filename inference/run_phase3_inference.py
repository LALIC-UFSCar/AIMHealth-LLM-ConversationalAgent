#!/usr/bin/env python3
"""
Phase 3 Inference Script: Persona-Aware RAG for Mental Health Chatbot

This script runs the full Phase 3 inference on the dataset with personas
and saves the results to an Excel file.

Usage:
    python -m inference.run_phase3_inference

Output:
    ./outputs/dataset_completo_com_personas_e_inferencias.xlsx
"""

import os
import sys
import json
import re
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# IMPORTS
# ============================================================================

print("=" * 70)
print("PHASE 3: PERSONA-AWARE RAG INFERENCE")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

print("\n📦 Loading libraries...")
import torch
import pandas as pd
from tqdm import tqdm
from inference.prompt_templates import build_phase3_prompt
from inference.guardrail import build_crisis_response_988, detect_crisis_risk
from config import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_OUTPUT_DIR,
    ensure_parent_dir,
    init_env,
    path_from_env,
)

print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# MENTAL HEALTH KEYWORDS
# ============================================================================

MENTAL_HEALTH_KEYWORDS = [
    'anxious', 'anxiety', 'panic', 'worried', 'stress', 'stressed', 'overwhelmed',
    'depressed', 'depression', 'sad', 'sadness', 'hopeless', 'hopelessness',
    'lonely', 'loneliness', 'isolated', 'alone', 'empty', 'numb',
    'scared', 'fear', 'afraid', 'terrified', 'nervous',
    'angry', 'frustrated', 'irritable', 'upset',
    'exhausted', 'tired', 'burnout', 'drained',
    'worthless', 'guilty', 'shame', 'self-esteem',
    'crying', 'cant sleep', 'insomnia', 'nightmares',
    'panic attack', 'heart racing', 'breathing', 'chest',
    'struggling', 'suffering', 'hurting', 'pain',
    'help', 'support', 'therapy', 'therapist', 'counseling',
    'feeling', 'emotions', 'emotional', 'mental health',
    'relationship', 'breakup', 'divorce', 'grief', 'loss', 'death',
    'job', 'work', 'school', 'exam', 'deadline', 'pressure',
    'family', 'friend', 'conflict'
]

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT_PHASE3 = """# Role: Compassionate Virtual Friend for Emotional Support

You are an AI companion specialized in providing empathetic emotional support. You combine the warmth of a close friend with evidence-based mental health support strategies.

## Core Identity
- You are an AI assistant, NOT a human - never claim personal experiences
- You are a supportive friend, NOT a clinician - avoid clinical language
- You provide emotional support backed by mental health research
- You do NOT have a name - never mention any name for yourself

## Communication Style

### Personalization
- Treat each person as a unique individual
- Use their name if provided
- Write in first person to create connection

### Language
- Gender-neutral, inclusive language always
- Casual and friendly tone (like a supportive friend)
- Use emojis sparingly 💙 (1-2 max, avoid during heavy moments)

### NEVER DO
- Give orders: "You must...", "Do this...", "Stop..."
- Claim experiences: "I also feel...", "I've been through..."
- Use clinical terms: "symptoms", "diagnosis", "treatment"
- Minimize: "it's not that bad", "just think positive"
- Mention that you are using a "persona" or "profile" file

### ALWAYS DO
- Gentle suggestions: "Some people find helpful...", "You might consider..."
- Validate first: "That sounds challenging", "It makes sense to feel that way"
- Ask with curiosity: "Can you tell me more?", "What was that like?"

## Response Framework (Think Step-by-Step)

Before responding, think through:
1. ACKNOWLEDGE: What emotion is the person sharing?
2. VALIDATE: How can I show this feeling makes sense?
3. EXPLORE: Should I ask questions or offer support?
4. RESPOND: Craft warm, empathetic response

## Knowledge Integration (CRITICAL FOR RAG)
When REFERENCE INFORMATION is provided:
1. **READ CAREFULLY** - Contains evidence-based guidance
2. **EXTRACT KEY INSIGHTS** - Identify practical advice, coping strategies
3. **INTEGRATE NATURALLY** - Weave insights into your empathetic response
4. **PRIORITIZE RELEVANCE** - Only use information that directly helps
5. **IGNORE IF IRRELEVANT** - If the reference doesn't match, rely on training

## Persona Context Integration (WHEN PROVIDED)
When PERSONA CONTEXT is provided:
1. Use it to understand the person's life situation and challenges
2. Tailor your response to their specific context
3. Reference their challenges with empathy, not assumption
4. DO NOT reveal that you have their profile or persona information
5. DO NOT stereotype based on any demographic information
6. DO NOT mention age, race, religion, or ethnicity explicitly"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_personas_from_json(json_path: str) -> Dict[str, Dict]:
    """Load persona definitions from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    personas_by_id = {}
    
    # AMIVE personas
    for persona in data.get('personas', {}).get('existing', []):
        pid = persona.get('id')
        if pid:
            personas_by_id[pid] = {
                'source': 'AMIVE',
                'demographics': persona.get('demographics', {}),
                'profile': persona.get('profile', {}),
                'keywords': persona.get('keywords', [])
            }
    
    # PersonaLens personas
    for persona in data.get('personas', {}).get('new', []):
        pid = persona.get('id')
        if pid:
            personas_by_id[pid] = {
                'source': 'PersonaLens',
                'demographics': persona.get('demographics', {}),
                'interests': persona.get('interests', {}),
                'affinities': persona.get('affinities', {}),
                'personality_summary': persona.get('personality_summary', '')
            }
    
    return personas_by_id


def extract_client_text(raw_input: str) -> str:
    """Extract client/user parts from dialogue."""
    if not raw_input or not isinstance(raw_input, str):
        return raw_input or ""
    
    client_pattern_brackets = r'\[Client\]\s*(.*?)(?=\[Therapist\]|\[Client\]|$)'
    matches_brackets = re.findall(client_pattern_brackets, raw_input, re.IGNORECASE | re.DOTALL)
    
    client_pattern_colon = r'(?:^|(?<=\n))Client:\s*(.*?)(?=(?:\n)?Therapist:|(?:\n)?Client:|$)'
    matches_colon = re.findall(client_pattern_colon, raw_input, re.IGNORECASE | re.DOTALL)
    
    all_matches = matches_brackets + matches_colon
    
    if all_matches:
        client_text = ' '.join(m.strip() for m in all_matches if m.strip())
        return client_text if client_text else raw_input
    
    return raw_input


def is_mental_health_related(text: str) -> bool:
    """Check if text is mental health related."""
    if not text:
        return False
    
    text_lower = text.lower()
    keyword_count = sum(1 for kw in MENTAL_HEALTH_KEYWORDS if kw in text_lower)
    
    first_person_patterns = [
        r"i('m| am| feel| have been| can't| don't)",
        r"my (anxiety|depression|stress|fear|worry)",
        r"i've been (feeling|struggling|dealing)",
    ]
    has_personal = any(re.search(p, text_lower) for p in first_person_patterns)
    
    return keyword_count >= 2 or (keyword_count >= 1 and has_personal)


def extract_age_group(persona: Dict) -> Optional[str]:
    """Extract age group from persona (SAFE)."""
    demographics = persona.get('demographics', {})
    source = persona.get('source', '')
    
    if source == 'AMIVE':
        age = demographics.get('age')
        if isinstance(age, (int, float)):
            if age < 18:
                return 'teen'
            elif age < 25:
                return 'young adult'
            elif age < 45:
                return 'adult'
            else:
                return 'older adult'
    
    elif source == 'PersonaLens':
        age_str = demographics.get('age', '')
        if '18-24' in age_str:
            return 'young adult'
        elif '25-34' in age_str or '35-44' in age_str:
            return 'adult'
        elif '45-54' in age_str:
            return 'adult'
        elif '55' in age_str or '65' in age_str:
            return 'older adult'
        elif '13' in age_str or '17' in age_str:
            return 'teen'
    
    return None


def extract_life_context(persona: Dict) -> Optional[str]:
    """Extract life context (student vs working) from persona (SAFE)."""
    demographics = persona.get('demographics', {})
    profile = persona.get('profile', {})
    source = persona.get('source', '')
    
    if source == 'AMIVE':
        education = demographics.get('education', '').lower()
        if any(term in education for term in ['year', 'student', 'university', 'college', 'master', 'phd']):
            return 'student'
        if 'working' in str(profile).lower():
            return 'working professional'
    
    elif source == 'PersonaLens':
        employment = demographics.get('employment_status', '').lower()
        if 'student' in employment:
            return 'student'
        elif 'working' in employment or 'full-time' in employment or 'part-time' in employment:
            return 'working professional'
        elif 'retired' in employment:
            return 'retired'
    
    return None


def build_augmented_query(user_text: str, persona: Dict, max_keywords: int = 6) -> str:
    """Build retrieval query with SAFE persona attributes."""
    augment_terms = []
    
    age_group = extract_age_group(persona)
    if age_group:
        augment_terms.append(age_group)
    
    life_context = extract_life_context(persona)
    if life_context:
        augment_terms.append(life_context)
    
    if persona.get('source') == 'AMIVE' and is_mental_health_related(user_text):
        keywords = persona.get('keywords', [])
        if keywords:
            input_lower = user_text.lower()
            relevant_keywords = [kw for kw in keywords if kw.lower() in input_lower]
            other_keywords = [kw for kw in keywords if kw not in relevant_keywords]
            
            selected = relevant_keywords[:max_keywords]
            if len(selected) < max_keywords and other_keywords:
                remaining = max_keywords - len(selected)
                selected.extend(random.sample(other_keywords, min(remaining, len(other_keywords))))
            
            augment_terms.extend(selected[:max_keywords])
    
    if augment_terms:
        augment_str = ' '.join(augment_terms)
        return f"{user_text} {augment_str}"
    
    return user_text


def persona_to_summary(persona: Dict) -> str:
    """Generate concise persona summary for prompt."""
    bullets = []
    source = persona.get('source', '')
    demographics = persona.get('demographics', {})
    
    age_group = extract_age_group(persona)
    life_context = extract_life_context(persona)
    
    if age_group and life_context:
        bullets.append(f"• {age_group.title()} who is a {life_context}")
    elif age_group:
        bullets.append(f"• {age_group.title()}")
    elif life_context:
        bullets.append(f"• {life_context.title()}")
    
    if source == 'AMIVE':
        profile = persona.get('profile', {})
        
        pains = profile.get('pains', '')
        if pains and isinstance(pains, str):
            pain_summary = pains[:150] + '...' if len(pains) > 150 else pains
            bullets.append(f"• Current challenges: {pain_summary}")
        
        clinical = profile.get('clinical_profile', '')
        if clinical:
            clinical_lower = clinical.lower()
            emotional_states = []
            if 'anxiety' in clinical_lower or 'anxious' in clinical_lower:
                emotional_states.append('experiences anxiety')
            if 'depression' in clinical_lower or 'depressive' in clinical_lower:
                emotional_states.append('experiences low mood')
            if 'stress' in clinical_lower:
                emotional_states.append('dealing with stress')
            if emotional_states:
                bullets.append(f"• Emotional context: {', '.join(emotional_states[:3])}")
        
        hobbies = profile.get('hobbies', [])
        if hobbies and isinstance(hobbies, list) and len(hobbies) > 0:
            hobbies_str = ', '.join(hobbies[:3])
            bullets.append(f"• Enjoys: {hobbies_str}")
    
    elif source == 'PersonaLens':
        personality = persona.get('personality_summary', '')
        if personality:
            safe_patterns = [
                r'\b(muslim|christian|jewish|hindu|buddhist|religious|non-religious)\b',
                r'\b(irish|british|austrian|asian|white|black)\b',
                r'\bfrom\s+(ireland|uk|austria|united kingdom)\b',
            ]
            clean_personality = personality
            for pattern in safe_patterns:
                clean_personality = re.sub(pattern, '', clean_personality, flags=re.IGNORECASE)
            clean_personality = re.sub(r'\s+', ' ', clean_personality).strip()
            
            if '. ' in clean_personality:
                first_sentence = clean_personality.split('. ')[0]
                if len(first_sentence) > 20:
                    bullets.append(f"• {first_sentence}")
        
        affinities = persona.get('affinities', {})
        if affinities:
            messaging = affinities.get('Messaging', {})
            comm_style = messaging.get('Preferred Communication Style', '')
            if comm_style:
                bullets.append(f"• Communication style: {comm_style}")
    
    bullets = bullets[:5]
    
    if not bullets:
        return "• General user seeking emotional support"
    
    return '\n'.join(bullets)


# ============================================================================
# MAIN INFERENCE FUNCTION
# ============================================================================

def generate_rag_response_with_persona(
    query: str,
    retriever,
    model,
    tokenizer,
    persona: Optional[Dict] = None,
    use_persona_in_retrieval: bool = True,
    use_persona_in_prompt: bool = True,
    max_new_tokens: int = 600,
    verbose: bool = False
) -> Tuple[str, List[Dict], Dict]:
    """Generate response using RAG + persona."""
    
    debug = {
        'persona_id': None,
        'persona_source': None,
        'use_persona_in_retrieval': use_persona_in_retrieval,
        'use_persona_in_prompt': use_persona_in_prompt,
        'clean_query': None,
        'augmented_query': None,
        'is_mental_health': None,
        'persona_summary': None,
        'is_crisis': False,
        'crisis_risk_level': 'NO_RISK',
        'crisis_signals': [],
        'crisis_path': 'normal',
    }
    
    # Step 1: Clean query
    clean_query = extract_client_text(query)
    debug['clean_query'] = clean_query[:200] + '...' if len(clean_query) > 200 else clean_query

    # Step 1.5: Crisis guardrail (highest priority)
    crisis = detect_crisis_risk(clean_query)
    debug['is_crisis'] = bool(crisis['is_crisis'])
    debug['crisis_risk_level'] = str(crisis['risk_level'])
    debug['crisis_signals'] = list(crisis['signals'])
    if crisis['is_crisis']:
        debug['crisis_path'] = 'early_return_988'
        return build_crisis_response_988(), [], debug
    
    # Step 2: Check if mental health related
    mh_related = is_mental_health_related(clean_query)
    debug['is_mental_health'] = mh_related
    
    # Step 3: Build retrieval query
    if use_persona_in_retrieval and persona is not None and mh_related:
        retrieval_query = build_augmented_query(clean_query, persona)
        debug['augmented_query'] = retrieval_query[:200]
        debug['persona_id'] = persona.get('demographics', {}).get('name') or 'PersonaLens'
        debug['persona_source'] = persona.get('source')
    else:
        retrieval_query = clean_query
        debug['augmented_query'] = None
        if persona:
            debug['persona_id'] = persona.get('demographics', {}).get('name') or 'PersonaLens'
            debug['persona_source'] = persona.get('source')
    
    # Step 4: Retrieve documents
    retrieved_docs = retriever.retrieve(retrieval_query)
    
    # Step 5: Format context
    context_parts = []
    for i, doc in enumerate(retrieved_docs[:3]):
        text = doc.get('text', '')
        if len(text) > 500:
            cut_point = 500
            for end_char in ['. ', '.\n', '? ', '!\n']:
                last_end = text[:500].rfind(end_char)
                if last_end > 200:
                    cut_point = last_end + 1
                    break
            text = text[:cut_point].strip()
        context_parts.append(f"[{i+1}]: {text}")
    context = "\n\n".join(context_parts)
    
    # Step 6: Build persona block
    persona_context = ""
    if use_persona_in_prompt and persona is not None:
        persona_summary = persona_to_summary(persona)
        debug['persona_summary'] = persona_summary
        persona_context = persona_summary
    
    # Step 7: Build prompt
    prompt = build_phase3_prompt(
        persona_context=persona_context,
        retrieved_context=context,
        user_message=clean_query,
    )
    
    # Step 8: Generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    input_length = inputs['input_ids'].shape[1]
    response_tokens = outputs[0][input_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
    
    return response, retrieved_docs, debug


# ============================================================================
# BATCH INFERENCE FUNCTION
# ============================================================================

def run_phase3_inference_batch(
    df: pd.DataFrame,
    retriever,
    model,
    tokenizer,
    personas_by_id: Dict,
    use_persona_in_retrieval: bool = True,
    use_persona_in_prompt: bool = True,
    max_new_tokens: int = 600,
    output_column: str = 'resposta_phase3'
) -> pd.DataFrame:
    """Run Phase 3 inference on a DataFrame batch."""
    df = df.copy()
    
    print(f"\n{'=' * 70}")
    print(f"PHASE 3 BATCH INFERENCE")
    print(f"{'=' * 70}")
    print(f"   Rows: {len(df)}")
    print(f"   use_persona_in_retrieval: {use_persona_in_retrieval}")
    print(f"   use_persona_in_prompt: {use_persona_in_prompt}")
    print("=" * 70)
    
    responses = []
    debug_infos = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🚀 Phase 3 Inference"):
        user_input = str(row['input']) if pd.notna(row['input']) else ""
        persona_id = row.get('persona', None)
        persona = personas_by_id.get(persona_id) if persona_id else None
        
        try:
            response, docs, debug = generate_rag_response_with_persona(
                query=user_input,
                retriever=retriever,
                model=model,
                tokenizer=tokenizer,
                persona=persona,
                use_persona_in_retrieval=use_persona_in_retrieval,
                use_persona_in_prompt=use_persona_in_prompt,
                max_new_tokens=max_new_tokens,
                verbose=False
            )
            responses.append(response)
            debug_infos.append(json.dumps(debug, ensure_ascii=False))
        except Exception as e:
            responses.append(f"[ERROR] {str(e)}")
            debug_infos.append(json.dumps({'error': str(e)}))
    
    df[output_column] = responses
    df[f'{output_column}_debug'] = debug_infos
    
    success_count = sum(1 for r in responses if not r.startswith('[ERROR]'))
    print(f"\n✅ Inference complete!")
    print(f"   Successful: {success_count}/{len(responses)}")
    
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def parse_args() -> argparse.Namespace:
    init_env()
    data_dir = path_from_env("PERSONARAG_DATA_DIR", DEFAULT_DATA_DIR)
    models_dir = path_from_env("PERSONARAG_MODELS_DIR", DEFAULT_MODELS_DIR)
    output_dir = path_from_env("PERSONARAG_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)

    parser = argparse.ArgumentParser(description="Phase 3 persona-aware RAG inference")
    parser.add_argument("--dataset", type=Path, default=data_dir / "dataset_completo_com_inferencias_final.xlsx")
    parser.add_argument("--personas", type=Path, default=data_dir / "all_personas_dataset.json")
    parser.add_argument("--output", type=Path, default=output_dir / "dataset_completo_com_inferencias_final.xlsx")
    parser.add_argument("--adapter-path", type=Path, default=models_dir / "llama-2-amive-adapter")
    parser.add_argument("--base-model", default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--es-host", default=os.getenv("PERSONARAG_ES_HOST", "http://localhost:9200"))
    parser.add_argument("--es-index", default=os.getenv("PERSONARAG_ES_INDEX", "oms_mental_health_docs"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=600)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Step 1: Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_excel(args.dataset)
    print(f"   Loaded {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    
    # Step 2: Load personas
    print("\n👤 Loading personas...")
    personas_by_id = load_personas_from_json(str(args.personas))
    print(f"   Loaded {len(personas_by_id)} personas: {list(personas_by_id.keys())}")
    
    # Step 3: Load model
    print("\n🤖 Loading LLaMA-2-13B-Amive model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer loaded")
    
    # Load base model with quantization
    print("   Loading base model (4-bit quantization)...")
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print("   ✅ Base model loaded")
    
    # Load LoRA adapter
    print("   Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(args.adapter_path))
    model.eval()
    print("   ✅ LoRA adapter loaded")
    
    # Step 4: Setup Elasticsearch retriever
    print("\n🔍 Setting up HybridRetriever...")
    from elasticsearch import Elasticsearch
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    class HybridRetriever:
        """Hybrid retriever with BM25 + Dense vectors + RRF fusion."""
        
        def __init__(self, 
                     es_client,
                     index_name: str = "oms_saude_mental_chunks",
                     embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                     top_k: int = 5):
            self.es = es_client
            self.index_name = index_name
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.top_k = top_k
        
        def retrieve(self, query: str) -> List[Dict]:
            """Retrieve documents using hybrid search."""
            # Get embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Hybrid query with RRF
            search_body = {
                "size": self.top_k,
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"text": {"query": query, "boost": 1.0}}},
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                        "params": {"query_vector": query_embedding}
                                    }
                                }
                            }
                        ]
                    }
                }
            }
            
            try:
                response = self.es.search(index=self.index_name, body=search_body)
                results = []
                for hit in response['hits']['hits']:
                    results.append({
                        'text': hit['_source'].get('text', ''),
                        'score': hit['_score'],
                        'chunk_id': hit['_id'],
                        'source': hit['_source'].get('source', 'unknown')
                    })
                return results
            except Exception as e:
                print(f"   ⚠️ Retrieval error: {e}")
                return []
    
    # Connect to Elasticsearch
    es = Elasticsearch([args.es_host])
    if not es.ping():
        print("   ❌ Elasticsearch not available. Please start Elasticsearch first.")
        print("   Run: sudo systemctl start elasticsearch")
        sys.exit(1)
    
    print("   ✅ Connected to Elasticsearch")
    
    retriever = HybridRetriever(
        es_client=es,
        index_name=args.es_index,
        embedding_model_name=args.embedding_model,
        top_k=args.top_k,
    )
    print("   ✅ HybridRetriever initialized")
    
    # Step 5: Run inference
    print("\n" + "=" * 70)
    print("🚀 STARTING PHASE 3 INFERENCE ON ALL 240 ROWS")
    print("=" * 70)
    
    df_result = run_phase3_inference_batch(
        df=df,
        retriever=retriever,
        model=model,
        tokenizer=tokenizer,
        personas_by_id=personas_by_id,
        use_persona_in_retrieval=True,
        use_persona_in_prompt=True,
        max_new_tokens=args.max_new_tokens,
        output_column='inferencia_persona_final'
    )
    
    # Step 6: Save results
    print(f"\n💾 Saving results to: {args.output}")
    ensure_parent_dir(args.output)
    df_result.to_excel(args.output, index=False)
    print(f"   ✅ Saved {len(df_result)} rows to Excel")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 INFERENCE COMPLETE")
    print("=" * 70)
    print(f"   Total rows: {len(df_result)}")
    print(f"   Output file: {args.output}")
    print(f"   Columns added: resposta_phase3, resposta_phase3_debug")
    
    # Show sample
    print("\n📋 Sample response (row 0):")
    print("-" * 50)
    sample_response = df_result.iloc[0]['resposta_phase3']
    print(sample_response[:500] + "..." if len(sample_response) > 500 else sample_response)
    
    print("\n" + "=" * 70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
