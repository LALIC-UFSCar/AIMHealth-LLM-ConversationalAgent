"""LLM-as-judge evaluator core."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import pandas as pd
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

from .rubric import SYSTEM_PROMPT, build_user_prompt
from .scoring import calc_nota_final, determine_melhor_saida

MAX_RETRIES = 5
INITIAL_BACKOFF = 2
REQUEST_TIMEOUT = 600


@dataclass
class EvaluatorConfig:
    model: str = "gpt-4o"
    temperature: float = 0.0


def _validate_result(result: dict) -> None:
    for key in ("response_1", "response_2", "response_3"):
        if key not in result:
            raise KeyError(f"Missing key '{key}' in evaluation result")
        sub = result[key]
        for criterion in ("empathy", "thematic_adequacy", "personalization"):
            if criterion not in sub:
                raise KeyError(f"Missing '{criterion}' in '{key}'")
            val = int(sub[criterion])
            if val not in (1, 2, 3):
                raise ValueError(f"Score for {key}.{criterion}={val}, must be 1,2,3")
            sub[criterion] = val


def evaluate_with_llm(client: OpenAI, cfg: EvaluatorConfig, persona_name: str, persona_description: str, user_input: str, response_1: str, response_2: str, response_3: str) -> dict:
    user_prompt = build_user_prompt(persona_name, persona_description, user_input, response_1, response_2, response_3)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            params = {
                "model": cfg.model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if not cfg.model.startswith("gpt-5"):
                params["temperature"] = cfg.temperature

            completion = client.chat.completions.create(**params)
            raw = completion.choices[0].message.content.strip()
            result = json.loads(raw)
            _validate_result(result)
            return result
        except (RateLimitError, APIConnectionError, APITimeoutError, APIError):
            if attempt == MAX_RETRIES:
                raise
            time.sleep(INITIAL_BACKOFF * (2 ** (attempt - 1)))
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(INITIAL_BACKOFF)

    raise RuntimeError("Evaluation failed after all retries")


def evaluate_dataframe(df: pd.DataFrame, persona_map: dict[str, str], cfg: EvaluatorConfig) -> pd.DataFrame:
    client = OpenAI(timeout=REQUEST_TIMEOUT)
    rows: list[dict] = []

    for _, row in df.iterrows():
        persona_name = str(row["persona"])
        persona_description = persona_map.get(persona_name, f"Persona: {persona_name}")

        eval_result = evaluate_with_llm(
            client=client,
            cfg=cfg,
            persona_name=persona_name,
            persona_description=persona_description,
            user_input=str(row["input"]),
            response_1=str(row["response_1"]),
            response_2=str(row["response_2"]),
            response_3=str(row["response_3"]),
        )

        r1 = eval_result["response_1"]
        r2 = eval_result["response_2"]
        r3 = eval_result["response_3"]

        nf1 = calc_nota_final(r1["empathy"], r1["personalization"], r1["thematic_adequacy"])
        nf2 = calc_nota_final(r2["empathy"], r2["personalization"], r2["thematic_adequacy"])
        nf3 = calc_nota_final(r3["empathy"], r3["personalization"], r3["thematic_adequacy"])

        rows.append(
            {
                "persona": persona_name,
                "id": row["id"],
                "dataset": row.get("dataset", ""),
                "input": row["input"],
                "empatia1": r1["empathy"],
                "personalizacao1": r1["personalization"],
                "adequacao1": r1["thematic_adequacy"],
                "empatia2": r2["empathy"],
                "personalizacao2": r2["personalization"],
                "adequacao2": r2["thematic_adequacy"],
                "empatia3": r3["empathy"],
                "personalizacao3": r3["personalization"],
                "adequacao3": r3["thematic_adequacy"],
                "notafinal1": nf1,
                "notafinal2": nf2,
                "notafinal3": nf3,
                "melhorsaida": determine_melhor_saida(nf1, nf2, nf3),
            }
        )

    out = pd.DataFrame(rows)
    out["winner_nota_final"] = out.apply(
        lambda x: max(("versao_1", x["notafinal1"]), ("versao_2", x["notafinal2"]), ("versao_3", x["notafinal3"]), key=lambda y: y[1])[0],
        axis=1,
    )
    return out
