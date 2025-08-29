from functools import wraps
import json
import os
from typing import List
from openai import OpenAI, api_key
from pprint import pprint
from difflib import unified_diff
from difflib import SequenceMatcher

import dotenv
from pydantic import BaseModel

dotenv.load_dotenv()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


class LLMEdit(BaseModel):
    lines: str
    content: str


class CanvasDiff(BaseModel):
    old_range: tuple[int, int]
    new_range: tuple[int, int]
    old_text: str
    new_text: str


def calculate_diff(old_content, new_content):
    # Create a matcher on the two strings
    matcher = SequenceMatcher(None, old_content, new_content, autojunk=False)
    # If there are two diffs "(Af)t(er)" that are less than merge_distance appart
    # merge them
    merge_distance = 1

    changes = []
    for tag, old_s, old_e, new_s, new_e in matcher.get_opcodes():
        if tag != "equal":
            # If there are two diffs "(Af)t(er)" that are less than merge_distance appart
            last_change = changes[-1] if changes else None
            if (
                last_change
                and len(new_content[last_change.new_range[1] : new_s]) <= merge_distance
            ):
                last_change.old_range = (last_change.old_range[0], old_e)
                last_change.new_range = (last_change.new_range[0], new_e)
                last_change.old_text = old_content[
                    last_change.old_range[0] : last_change.old_range[1]
                ]
                last_change.new_text = new_content[
                    last_change.new_range[0] : last_change.new_range[1]
                ]
            else:
                changes.append(
                    CanvasDiff(
                        old_range=(old_s, old_e),
                        new_range=(new_s, new_e),
                        old_text=old_content[old_s:old_e],
                        new_text=new_content[new_s:new_e],
                    )
                )

    return changes


def modifies_canvas(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        old_content = self.canvas_data.content
        result = func(self, *args, **kwargs)
        new_content = self.canvas_data.content

        if old_content != new_content:
            self.canvas_data.changes = calculate_diff(old_content, new_content)
        return result

    return wrapper


class CanvasData(BaseModel):
    content: str
    changes: List[CanvasDiff]


class CanvasLLM:
    client: OpenAI
    canvas_data: CanvasData

    def __init__(self, canvas_content: str = "", client: OpenAI | None = None):
        self.client = (
            OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            if client is None
            else client
        )
        self.canvas_data = CanvasData(content=canvas_content, changes=[])

    def invoke(self, messages):
        return self.client.chat.completions.create(model="gpt-4o", messages=messages)

    def generate_canvas(self, instructions) -> dict:
        response = self.invoke(
            [
                {
                    "role": "user",
                    "content": f"You are a canvas writing app. You will be given instructions by the user, return ONLY the content that the user requested and no other information intro, outro or any chat fillers. Instructions: {instructions}",
                }
            ],
        )
        self.canvas_data.content = response.choices[0].message.content
        return self.canvas_data.content

    @modifies_canvas
    def apply_edits(self, edits: List[LLMEdit]):
        for edit in sorted(
            edits, key=lambda e: int(e.lines.split("-")[0]), reverse=True
        ):
            lines = edit.lines.split("-")
            start = int(lines[0]) - 1
            end = int(lines[1]) - 1
            canvas_lines = self.canvas_data.content.split("\n")
            self.canvas_data.content = "\n".join(
                canvas_lines[:start]
                + edit.content.split("\n")
                + canvas_lines[(end + 1) :]
            )

    def edit_canvas_numbered(self, instructions):
        numbered_canvas_content = self.canvas_data.content

        numbered_canvas_content = "\n".join(
            [f"{i+1}: {x}" for i, x in enumerate(numbered_canvas_content.split("\n"))]
        )

        response = self.invoke(
            [
                {
                    "role": "user",
                    "content": f"""You are a canvas editing app. You will be given instructions by the user, return ONLY the content that the user requested and no other information. 

Instructions from the user for editing the content:
<instructions>
{instructions}
</instructions>

You have to edit the following 1 based numbered lines of canvas content:
<canvas_content>
{numbered_canvas_content}
</canvas_content>

Reply with the edits that you would like to make to the canvas content. Reply in JSON in the following format:
```json
{{
    "edits": [
        {{
            "lines": "1-1",
            "content": "New content for the first line"
        }},
        {{
            "lines": "2-3",
            "content": "New content for the second and \n third line"
        }}
    ]
}}
```

Do not include any additional context or explanations, only the json.
""",
                }
            ],
        )

        json_string = json.loads(
            response.choices[0].message.content.strip("```json\n").strip("```")
        )

        return self.apply_edits([LLMEdit(**edit) for edit in json_string["edits"]])

    @modifies_canvas
    def edit_canvas_rewrite(self, instructions):
        response = self.invoke(
            [
                {
                    "role": "user",
                    "content": f"""You are a canvas editing app. You will be given instructions by the user, return ONLY the content that the user requested and no other information. 

Instructions from the user for editing the content:
<instructions>
{instructions}
</instructions>

You have to edit the following lines of canvas content:
<canvas_content>
{self.canvas_data.content}
</canvas_content>

Reply with the new content of the canvas. Change only what is necessary to fulfill the user's instructions.
Do not include any additional context or explanations, only the new content. Wrap the content in <canvas_content> tags.
""",
                }
            ],
        )

        self.canvas_data.content = (
            response.choices[0]
            .message.content.strip("<canvas_content>\n")
            .strip("\n</canvas_content>")
        )
        return self.canvas_data.content


def print_canvas_with_changes(canvas: CanvasData):
    """This function prints the canvas to terminal, the changes should have a blue background"""

    for i, char in enumerate(canvas.content):
        if any([x.new_range[0] <= i < x.new_range[1] for x in canvas.changes]):
            print(f"\033[44m{char}\033[0m", end="")
        else:
            print(char, end="")
    print()


if __name__ == "__main__":
    canvas = CanvasLLM()

    canvas.generate_canvas(input("Instructions to generate the initial content:\n> "))
    print_canvas_with_changes(canvas.canvas_data)

    while True:
        instructions = input("\n> ")
        print()
        canvas.edit_canvas_numbered(instructions)
        print_canvas_with_changes(canvas.canvas_data)
