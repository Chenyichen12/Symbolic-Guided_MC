import re
from typing import Any, Dict, List, Optional, Union


def parse_leaf_node_value(response: str, gold_answer: str):
    matched_correct = 0
    matched_text = re.findall(r'\[.*?\]', response)
    if len(matched_text) > 0:
        predicted_answer = matched_text[-1].lower()
        predicted_answer = predicted_answer[1:-1]
    elif "answer:" in response:
        predicted_answer = response.lower().split("answer: ")[-1]
    else:
        predicted_answer = response.split("\n")[-1].lower()
    
    predicted_label = "unknown"
    if predicted_answer == "a":
       predicted_label = "A"
    elif predicted_answer == "b":
       predicted_label = "B"
    elif predicted_answer == "c":
       predicted_label = "C"
    elif predicted_answer == "no":
       predicted_label = "B"
    elif "true" in predicted_answer:
       predicted_label = "A"
    elif "false" in predicted_answer:
       predicted_label = "B"
    elif "yes" in predicted_answer:
       predicted_label = "A"
    elif "uncertain" in predicted_answer:
       predicted_label = "C"
    elif "no" in predicted_answer.split():
       predicted_label = "B"
    elif "is not" in predicted_answer:
       predicted_label = "unknown"
    elif "is no" in predicted_answer:
       predicted_label = "B"
    else:
       predicted_label = "C"
      
    if gold_answer == predicted_label:
       matched_correct = 1
    elif gold_answer == "Uncertain" and predicted_label == "C":
       matched_correct = 1
    elif gold_answer == "True" and predicted_label == "A":
       matched_correct = 1
    elif gold_answer == "False" and predicted_label == "B":
       matched_correct = 1

    return matched_correct, predicted_label


def get_steps(text: str) -> List[str]:
    temp_steps = text.split('\n')
    final_steps = []
    step_cache = []
    action = False
    for step in temp_steps:
        if len(step_cache) > 0:
            if step.startswith("Thought:"):
                final_steps.append('\n'.join(step_cache))
                step_cache = []
                action = False
            
            if step.startswith("Action:") and action:
                final_steps.append('\n'.join(step_cache))
                step_cache = []
                action = False
      
        step_cache.append(step)
        if step.startswith("Action:"):
            action = True
      
    if len(step_cache) > 0:
        final_steps.append('\n'.join(step_cache))
    
    return final_steps