"""
A module for testing LLM edit accuracy.
"""

import os
import random
from typing import Callable
from click import prompt
import nltk
from openai import OpenAI
import __main__
from tqdm import tqdm

from main import CanvasLLM


def distance(text1: str, text2: str) -> float:
    """
    Compute the normalized levenstein distance between two texts.
    """
    return 1 - nltk.edit_distance(text1, text2) / max(len(text1), len(text2))


def replace_words_test(
    original_text: str, change_fn: Callable[[str, str], str], num_words: int = 3
) -> str:
    words = original_text.split()
    if len(words) < num_words:
        return (-1, [], "")

    random.shuffle(words)
    random_words = words[:num_words]
    random.shuffle(words)
    random_new_words = words[:num_words]

    target_text = original_text

    for new_word, word in zip(random_new_words, random_words):
        target_text = target_text.replace(word, new_word)

    prompt_word_replacements = [
        f"Replace '{word}' with '{new_word}'"
        for word, new_word in zip(random_words, random_new_words)
    ]
    prompt = (
        "Replace the following words in the text in the order specified (just replace the words, do not change anything else): "
        f"{"\n".join(prompt_word_replacements)}"
    )

    output = change_fn(prompt, original_text)
    return (
        distance(target_text, output),
        list(zip(random_new_words, random_words)),
        output,
    )


def replace_lines_test(
    original_text: str, change_fn: Callable[[str, str], str], num_lines: int = 1
) -> str:
    lines = [x for x in original_text.split("\n") if len(x.strip()) > 0]
    if len(lines) < num_lines:
        return (-1, [], "")

    random.shuffle(lines)
    random_lines = lines[:num_lines]
    random.shuffle(lines)
    random_new_lines = lines[:num_lines]

    target_text = original_text

    for new_line, line in zip(random_new_lines, random_lines):
        target_text = target_text.replace(line, new_line)

    prompt_line_replacements = [
        f"Replace '{line}' with '{new_line}'"
        for line, new_line in zip(random_lines, random_new_lines)
    ]
    prompt = (
        "Replace the following lines in the text (just replace the lines, do not change anything else): "
        f"{"\n".join(prompt_line_replacements)}"
    )

    output = change_fn(prompt, original_text)
    return (
        distance(target_text, output),
        list(zip(random_new_lines, random_lines)),
        output,
    )


def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def replace_numbers_test(
    original_text: str, change_fn: Callable[[str, str], str], num_numbers: int = 3
) -> str:
    numbers = [word for word in original_text.split() if is_float(word)]
    if len(numbers) < num_numbers:
        return (-1, [], "")

    random.shuffle(numbers)
    random_numbers = numbers[:num_numbers]
    random_new_numbers = [
        str(int(float(num) + random.randint(-100, 100))) for num in random_numbers
    ]

    target_text = original_text

    for new_number, number in zip(random_new_numbers, random_numbers):
        target_text = target_text.replace(number, new_number)

    prompt_number_replacements = [
        f"Replace '{number}' with '{new_number}'"
        for number, new_number in zip(random_numbers, random_new_numbers)
    ]
    prompt = (
        "Replace the following numbers in the text in the order specified(just replace the numbers, do not change anything else): "
        f"{"\n".join(prompt_number_replacements)}"
    )

    output = change_fn(prompt, original_text)
    return (
        distance(target_text, output),
        list(zip(random_numbers, random_new_numbers)),
        output,
    )


"""
interface for testing LLMs. Each test candidate should implement this interface.
"""


class LLMTestCandidate:
    def edit_canvas(self, instructions: str, canvas_content: str) -> str: ...
    def name(self) -> str: ...


def generate_initial_canvas(client: OpenAI, instructions: str) -> str:
    return CanvasLLM(client=client).generate_canvas(instructions)


def run_tests(client, candidates: list[LLMTestCandidate], iterations: int = 5):
    test_results_all = {}

    initial_text = (
        generate_initial_canvas(
            client,
            "Create an example subsection of a financial report, include some numbers",
        )
        .replace("$", "")
        .replace("€", "")
        .replace("%", "")
    )

    progress_bar = tqdm(total=len(candidates) * iterations * 3)
    progress_bar.set_description("Running tests")

    for candidate in candidates:

        test_results = {
            "replace_words": [],
            "replace_lines": [],
            "replace_numbers": [],
        }

        for _ in range(iterations):
            test_results["replace_words"].append(
                replace_words_test(initial_text, candidate.edit_canvas)
            )
            progress_bar.update(1)

            test_results["replace_lines"].append(
                replace_lines_test(initial_text, candidate.edit_canvas)
            )
            progress_bar.update(1)

            test_results["replace_numbers"].append(
                replace_numbers_test(initial_text, candidate.edit_canvas)
            )
            progress_bar.update(1)

        test_results_all[candidate.name()] = test_results

    return test_results_all


def print_results(results):
    tests = results[next(iter(results))].keys()
    for test in tests:
        print(f"{test}:")
        for candidate_name, candidate_results in results.items():
            candidate_test_results = candidate_results[test]
            avg_score = sum([score for score, _, _ in candidate_test_results]) / len(
                candidate_test_results
            )
            min_score = min([score for score, _, _ in candidate_test_results])
            max_score = max([score for score, _, _ in candidate_test_results])
            print(
                f"  {candidate_name}: {avg_score:.2f} [{min_score:.2f}, {max_score:.2f}]"
            )
        print()
    print("Candidate totals:")
    for candidate_name, candidate_results in results.items():
        avg_avg = 0
        avg_min = 0
        avg_max = 0
        for test in tests:
            candidate_test_results = candidate_results[test]
            avg_score = sum([score for score, _, _ in candidate_test_results]) / len(
                candidate_test_results
            )
            min_score = min([score for score, _, _ in candidate_test_results])
            max_score = max([score for score, _, _ in candidate_test_results])
            avg_avg += avg_score
            avg_min += min_score
            avg_max += max_score
        avg_avg /= len(tests)
        avg_min /= len(tests)
        avg_max /= len(tests)
        print(f"  {candidate_name}: {avg_avg:.2f} [{avg_min:.2f}, {avg_max:.2f}]")


if __name__ == "__main__":
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    class LineNumberEditor(LLMTestCandidate):
        def edit_canvas(self, instructions: str, canvas_content: str) -> str:
            canvas = CanvasLLM(canvas_content)
            canvas.edit_canvas_numbered(instructions)
            return canvas.canvas_data.content

        def name(self) -> str:
            return "Line Number Editor"

    class RewriteEditor(LLMTestCandidate):
        def edit_canvas(self, instructions: str, canvas_content: str) -> str:
            canvas = CanvasLLM(canvas_content)
            canvas.edit_canvas_rewrite(instructions)
            return canvas.canvas_data.content

        def name(self) -> str:
            return "Rewrite Editor"

    candidates = [
        LineNumberEditor(),
        RewriteEditor(),
    ]

    results = run_tests(client, candidates, iterations=5)
    print_results(results)

    if not hasattr(__main__, "__file__"):
        print("You can view the results in the 'results' variable.")
